import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
import math
from typing import Dict, List, Optional, Union
from functools import partial

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.api.task
import lm_eval.models
import torch.distributed as dist
import yaml
from lm_eval import evaluator as pretrain_evaluator
from lm_eval import utils
from lm_eval._cli.utils import check_argument_types
from lm_eval.api.model import LM
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import TaskManager as PretrainTaskManager
from lm_eval.utils import sanitize_model_name, simple_parse_args_string
from lm_eval.utils import handle_non_serializable as _orig_handle
from lm_eval.api.registry import get_model

# from eval.chat_benchmarks.curator_lm import CuratorAPIModel  # register curator model
from eval.chat_benchmarks.precomputed_hf_lm import PrecomputedHFLM  # register precomputed_hf model
from eval.chat_benchmarks.upload_to_hf_lm import UploadInstancesToHF  # register upload_to_hf model
from eval.constants import LIST_OPENAI_MODELS
from eval.eval_tracker import DCEvaluationTracker
from eval.task import TaskManager as InstructTaskManager
from eval.utils import eval_logger_fn
import traceback

eval_logger = eval_logger_fn()


_BIT_CAP = 15_000


def _int_or_none_list_arg_type(max_len: int, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items != max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )

    return items


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=dict,
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3),
        default="0,1234,1234",  # for backward compatibility
        help=(
            "Set seed for python's random, numpy and torch.\n"
            "Accepts a comma-separated list of 3 values for python's random, numpy, and torch seeds, respectively, "
            "or a single integer to set the same seed for all three.\n"
            "The values are either an integer or 'None' to not set the seed. Default is `0,1234,1234` (for backward compatibility).\n"
            "E.g. `--seed 0,None,8` sets `random.seed(0)` and `torch.manual_seed(8)`. Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all three seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )

    return parser



def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()


def handle_non_serializable_extended(o):
    """
    Delegates to the stock helper, but for gigantic SymPy Integer /
    Rational objects returns a short placeholder *without* calling str().
    """
    try:
        from sympy import Integer, Rational

        if isinstance(o, Integer):
            if o.p.bit_length() > _BIT_CAP:
                digits = int(o.p.bit_length() * math.log10(2)) + 1
                return f"<Integer ~{digits} digits>"
            return str(int(o))  # safe: fits under the guard

        if isinstance(o, Rational):
            num_bits = o.p.bit_length()
            den_bits = o.q.bit_length()
            if num_bits > _BIT_CAP or den_bits > _BIT_CAP:
                d_num = int(num_bits * math.log10(2)) + 1
                d_den = int(den_bits * math.log10(2)) + 1
                return f"<Rational {d_num}/{d_den} digits>"
            return str(o)  # small enough
    except ModuleNotFoundError:
        pass

    # Everything else: NumPy ints, sets, etc.
    return _orig_handle(o)


def setup_custom_parser():
    """
    Create a custom argument parser that extends lm-eval-harness parser.
    """
    parser = setup_parser()
    db_group = parser.add_argument_group("database")

    db_group.add_argument("--model_id", type=str, default=None, help="Model UUID for direct database tracking")

    parser.add_argument(
        "--use_database", action="store_true", help="Where to use PostgreSQL Database to track results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for direct database tracking. If not set, the model path will be used instead.",
    )
    db_group.add_argument(
        "--overwrite-database",
        action="store_true",
        help="By default, we do not overwrite database entry, but if this is passed, we will compute eval even if found in database.",
    )

    db_group.add_argument(
        "--is_external_model",
        action="store_true",
        help="By default, the model is stored as internal in the database. If set, this is overwritten to external.",
    )

    parser.add_argument(
        "--creation_location",
        type=str,
        default="NA",
        help="Specifies which compute server is used for evaluating the model.",
    )

    parser.add_argument(
        "--created_by",
        type=str,
        default="NA",
        help="Specifies who evaluates the model.",
    )

    parser.add_argument(
        "--annotator_model",
        type=str,
        default="auto",
        help="Judge model used to evaluate generations. Example: gpt-4o-mini-2024-07-18",
    )
    parser.add_argument(
        "--max_tokens",
        type=str,
        default=None,
        help="Maximum length of model generatd tokens.",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml. Overwrites --batch_size, --tasks, --annotator_model, and --max_tokens",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run evalutaions in debug mode on a few examples",
    )

    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to use for all tasks that support it.",
    )

    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Whether to apply chat template when constructing few-shot examples.",
    )

    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        help="Whether to format few-shot examples as multi-turn conversations when applying chat template.",
    )
    return parser


def evaluate(
    lm: LM,
    task_manager: InstructTaskManager,
    pretrain_task_manager: PretrainTaskManager,
    task_list: List[str],
    batch_sizes_list: List[int],
    verbosity: str = "INFO",
    args=None,
    **eval_kwargs,
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.

    Args:
        lm (LM):
            Language model instance to evaluate.
        task_manager (InstructTaskManager):
            Manager for instruction-based evaluation tasks.
        pretrain_task_manager (PretrainTaskManager):
            Manager for pre-training evaluation tasks.
        task_list (List[str]):
            List of task names to evaluate the model on.
        batch_sizes_list (List[int]):
            List of batch sizes for each task.
        verbosity (str, optional):
            Logging verbosity level. Defaults to "INFO".
        args (Any, optional):
            Additional arguments to pass to the evaluation. Defaults to None.
        **eval_kwargs:
            Additional keyword arguments for evaluation configuration.

    Returns:
        Dict[str, Dict]:
            Dictionary mapping task names to their evaluation results.
            Each result dictionary contains metrics specific to that task.
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    # Split tasks between benchmark and pretrain
    benchmark_tasks = [t for t in task_list if t in task_manager.tasks]
    benchmark_batch_sizes = [b for (t, b) in zip(task_list, batch_sizes_list) if t in task_manager.tasks]
    pretrain_tasks = [t for t in task_list if t in pretrain_task_manager.all_tasks]
    pretrain_batch_sizes = [b for (t, b) in zip(task_list, batch_sizes_list) if t in pretrain_task_manager.all_tasks]

    unknown_tasks = set(task_list).difference(set(benchmark_tasks)).difference(set(pretrain_tasks))

    if len(unknown_tasks) > 0:
        raise ValueError(f"Tasks {unknown_tasks} are not recognized.")

    if benchmark_tasks:
        eval_logger.info(f"Benchmark tasks to evaluate: {benchmark_tasks}")
    if pretrain_tasks:
        eval_logger.info(f"Pretrain tasks to evaluate: {pretrain_tasks}")

    results = {"results": {}}

    # Run benchmark evaluations - sequential generation, parallel evaluation
    if benchmark_tasks:
        # Sequential generation since it's GPU-bound
        generate_methods = task_manager.get_list_generate_responses(benchmark_tasks)
        generation_results = []
        valid_tasks = []  # Keep track of valid tasks
        for method, task, batch_size in zip(generate_methods, benchmark_tasks, benchmark_batch_sizes):
            if args.model == "hf":
                lm.batch_size_per_gpu = batch_size
            elif args.model == "vllm":
                lm.batch_size = batch_size
            result = method(lm)
            if result is not None:  # Only keep valid results and their corresponding tasks
                generation_results.append(result)
                valid_tasks.append(task)
        # Get evaluation methods only for valid tasks

        if lm is not None and not hasattr(lm, "upload_to_hub"):
            evaluate_methods = task_manager.get_list_evaluates(valid_tasks)
            cpu_count = os.cpu_count()

            max_workers = min(len(valid_tasks), cpu_count * 2)
            if lm.world_size <= 1 or lm.rank == 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    evaluate_results = list(
                        executor.map(
                            lambda func_args: func_args[0](func_args[1]), zip(evaluate_methods, generation_results)
                        )
                    )

                # Store results using valid tasks for correct mapping
                for task, result in zip(valid_tasks, evaluate_results):
                    results["results"][task] = result

    # Run pretrain evaluations if any exist
    if pretrain_tasks and args is not None:
        try:
            for pretrain_task, batch_size in zip(pretrain_tasks, pretrain_batch_sizes):
                pretrain_results = pretrain_evaluator.simple_evaluate(
                    model=args.model,
                    model_args=args.model_args,
                    tasks=[pretrain_task],
                    num_fewshot=args.num_fewshot,
                    batch_size=batch_size,
                    max_batch_size=args.max_batch_size,
                    device=args.device,
                    use_cache=args.use_cache,
                    limit=args.limit,
                    check_integrity=args.check_integrity,
                    write_out=args.write_out,
                    log_samples=args.log_samples,
                    evaluation_tracker=args.evaluation_tracker if hasattr(args, "evaluation_tracker") else None,
                    system_instruction=args.system_instruction,
                    apply_chat_template=args.apply_chat_template,
                    fewshot_as_multiturn=args.fewshot_as_multiturn,
                    gen_kwargs=args.gen_kwargs,
                    task_manager=pretrain_task_manager,
                    verbosity=args.verbosity,
                    predict_only=args.predict_only,
                    random_seed=args.seed[0] if hasattr(args, "seed") else None,
                    numpy_random_seed=args.seed[1] if hasattr(args, "seed") else None,
                    torch_random_seed=args.seed[2] if hasattr(args, "seed") else None,
                    fewshot_random_seed=args.seed[3] if hasattr(args, "seed") else None,
                )
                if pretrain_results is not None:
                    results["results"].update(pretrain_results.get("results", {}))
        except Exception as e:
            eval_logger.error(f"Error in pretrain evaluation: {str(e)}")

    # If we're using UploadInstancesToHF, make sure to call upload_to_hub
    if lm is not None and hasattr(lm, "upload_to_hub") and callable(lm.upload_to_hub):
        try:
            eval_logger.info("Uploading accumulated instances to HuggingFace Hub...")
            lm.upload_to_hub()
        except Exception as e:
            eval_logger.error(f"Error uploading instances to HF: {str(e)}")
            import traceback

            traceback.print_exc()

    # If we're using PrecomputedHFLM, update the README with evaluation results
    if lm is not None and hasattr(lm, "update_repo_readme") and callable(lm.update_repo_readme):
        try:
            eval_logger.info("Updating repository README with evaluation results...")
            local_readme_path = os.path.join(
                args.output_path, args.model_args.strip("repo_id=").replace("/", "__") + "_README.md"
            )
            lm.update_repo_readme(results, local_readme_path=local_readme_path)
        except Exception as e:
            eval_logger.error(f"Error updating repository README: {str(e)}")
            import traceback

            traceback.print_exc()

    return results


def update_model_args_with_name(model_args: str, model_name: str) -> str:
    """
    Update model_args string to include pretrained model name if not already present.

    Args:
        model_args: Original model args string
        model_name: Model name to add

    Returns:
        str: Updated model args string
    """
    if not model_args:
        return f"pretrained={model_name}"

    args_dict = simple_parse_args_string(model_args)
    if "pretrained" not in args_dict:
        return f"pretrained={model_name},{model_args}"
    else:
        assert (
            args_dict["pretrained"] == model_name
        ), f"Provided model_args contains different pretrained model '{args_dict['pretrained']}' than specified model_name '{model_name}'"
    return model_args


def cli_evaluate(args: Optional[argparse.Namespace] = None) -> None:
    """
    Command-line interface for evaluating language models.

    Args:
        args: Command line arguments. If None, will parse from sys.argv
    """
    # Parse arguments if not provided
    if not args:
        parser = setup_custom_parser()
        args = parse_eval_args(parser)

    if args.config is not None:
        # This overwrites `--tasks` and `--batch_size`
        with open(args.config, "r") as file:
            tasks_yaml = yaml.safe_load(file)
        args.tasks = ",".join([t["task_name"] for t in tasks_yaml["tasks"]])
        batch_sizes_list = [int(t["batch_size"]) if t["batch_size"] != "auto" else "auto" for t in tasks_yaml["tasks"]]
        args.annotator_model = tasks_yaml.get("annotator_model", args.annotator_model)
        args.max_tokens = int(tasks_yaml.get("max_tokens", args.max_tokens))
    else:
        batch_sizes_list = [
            int(args.batch_size) if args.batch_size != "auto" else args.batch_size
            for _ in range(len(args.tasks.split(",")))
        ]

    # # Initialize evaluation tracker
    # if args.output_path:
    #     args.hf_hub_log_args += f",output_path={args.output_path}"
    evaluation_tracker = setup_evaluation_tracker(args.output_path, args.use_database)

    task_list = args.tasks.split(",")

    # If model_id is provided, lookup model weights location from database
    if args.model_id:
        if not args.use_database:
            raise ValueError("--use_database must be set to use --model_id.")
        try:
            model_name = evaluation_tracker.get_model_attribute_from_db(args.model_id, "weights_location")
            args.model_args = update_model_args_with_name(args.model_args or "", model_name)
            eval_logger.info(f"Retrieved model name from database: {model_name}")
        except Exception as e:
            eval_logger.error(f"Failed to retrieve model name from database: {str(e)}")
            sys.exit(1)
        if not args.overwrite_database:
            task_list = [
                task for task in task_list if not evaluation_tracker.check_if_already_done(task, args.model_id)
            ]
            if len(task_list) == 0:
                eval_logger.info("All tasks passed in were found in the database.")
                exit()
    elif args.model_name:
        model_name = args.model_name
        args.model_args = update_model_args_with_name(args.model_args or "", model_name)

    # Initialize tasks
    task_manager = InstructTaskManager(
        annotator_model=args.annotator_model,
        max_tokens=int(args.max_tokens) if args.max_tokens else None,
        debug=args.debug,
        seed=args.seed,
        task_list=task_list,
        system_instruction=args.system_instruction,
    )
    pretrain_task_manager = PretrainTaskManager(args.verbosity, include_path=args.include_path)

    eval_logger.info(f"Selected Tasks: {[task for task in task_list]}")

    # Only check for OpenAI API keys if at least one task requires an annotator model
    # TODO: Should we just skip the evaluation that requires the annotator model if the annotator model is not set or fail completely?
    if args.annotator_model in LIST_OPENAI_MODELS and any(
        task_manager.requires_annotator_model(task) for task in task_list
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                f"Please set OPENAI_API_KEY to allow usage of {args.annotator_model}"
                f"to evaluate the following tasks: {[task for task in task_list if task_manager.requires_annotator_model(task)]}"
            )

    # Check if any task is not in either task manager
    if any(task not in task_manager.tasks and task not in pretrain_task_manager.all_tasks for task in task_list):
        raise ValueError(
            f"The following tasks could not be found: {[task for task in task_list if task not in task_manager.tasks and task not in pretrain_task_manager.all_tasks]}"
        )

    # Initialize model
    try:
        lm = initialize_model(args.model, args.model_args, batch_size=args.batch_size)
    except Exception as e:
        traceback.print_exc()
        eval_logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

    # Log experiment configuration
    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=args.model,
            model_args=args.model_args,
            system_instruction=args.system_instruction,
            chat_template=lm.chat_template(args.apply_chat_template),
            fewshot_as_multiturn=args.fewshot_as_multiturn,
        )

    # Initialize logging and environment
    
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Setup wandb logging if requested
    wandb_logger = None
    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # Run evaluation
    results = evaluate(
        lm=lm,
        task_manager=task_manager,
        pretrain_task_manager=pretrain_task_manager,
        task_list=task_list,
        batch_sizes_list=batch_sizes_list,
        verbosity=args.verbosity,
        args=args,
    )

    # Add metadata to results
    if lm.rank == 0:
        add_results_metadata(results, batch_sizes_list, args, lm)
        handle_evaluation_output(results, args, evaluation_tracker, wandb_logger)

    if dist.is_initialized():
        dist.destroy_process_group()


def setup_evaluation_tracker(output_path: str, use_database: bool) -> DCEvaluationTracker:
    """
    This function initializes a DCEvaluationTracker instance with the specified
    configuration for either file-based or database storage of evaluation results.

    Args:
        output_path (str): The file system path where evaluation results will be saved.
            For file-based storage, this will be the directory path. For database
            storage, this could be the connection string or database path.
        use_database (bool): If True, uses database storage for results.
            If False, uses file-based storage.

    Returns:
        DCEvaluationTracker: A configured instance of the evaluation tracker
            ready to record and manage DCF evaluation results
    """
    return DCEvaluationTracker(output_path, use_database)


def initialize_model(
    model: Union[str, LM],
    model_args: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> LM:
    """
    Initialize the language model based on provided configuration.

    Args:
        model (Union[str, LM]):
            Either a string identifier for the model to load from registry,
            or an already instantiated LM object.
        model_args (Optional[str], optional):
            Additional arguments for model initialization as a string.
            Only used if model is provided as a string. Defaults to None.
        device (Optional[str], optional):
            Device to load the model on (e.g., 'cuda', 'cpu'). Defaults to None.

    Returns:
        LM:
            Initialized language model instance with configured parameters
            and a sanitized model identifier.
    """
    if isinstance(model, str):
        if model_args is None:
            model_args = ""

        config = {
            "device": device,
        }

        if "batch_size" not in model_args:
            if batch_size is not None:
                model_args += f",batch_size={batch_size}"

        lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            config,
        )
        setattr(lm, "model_name", model)
    else:
        lm = model

    lm.model_identifier = sanitize_model_name(f"model_{model}_model_args_{model_args}")
    return lm


def add_results_metadata(results: Dict, batch_sizes_list: List[int], args: argparse.Namespace, lm: LM) -> None:
    """
    Add metadata and configuration to results.

    Args:
        results (Dict):
            Dictionary of evaluation results to be augmented with metadata.
            The function will modify this dictionary in-place to add
            configuration and runtime information.
        batch_sizes_list (List[int]):
            List of batch sizes for each task.
        args (argparse.Namespace):
            Command line arguments containing runtime configuration
            and parameters used during evaluation.
        lm (LM):
            Language model instance, used to extract model-specific
            configuration and parameters.

    Returns:
        None:
            The function modifies the results dictionary in-place.
    """
    results["config"] = {
        "model": (
            args.model
            if isinstance(args.model, str)
            else args.model.config._name_or_path
            if hasattr(args.model, "config")
            else type(args.model).__name__
        ),
        "model_args": args.model_args,
        "tasks": args.tasks,
        "batch_sizes": batch_sizes_list,
        "device": args.device,
        "use_cache": args.use_cache,
        "limit": args.limit,
        "annotator_model": args.annotator_model,
        "max_tokens": args.max_tokens if args.max_tokens is not None else "default",
        # "bootstrap_iters": args.bootstrap_iters,
        "gen_kwargs": args.gen_kwargs,
        "random_seed": args.seed[0],
        "numpy_seed": args.seed[1],
        "torch_seed": args.seed[2],
        # "fewshot_seed": args.seed[3],
    }

    if isinstance(lm, get_model("huggingface")):
        results["config"].update(lm.get_model_info())

    results["git_hash"] = get_git_commit_hash()
    results["date"] = time.time()
    add_env_info(results)
    add_tokenizer_info(results, lm)


def handle_evaluation_output(
    results: Dict,
    args: argparse.Namespace,
    evaluation_tracker: EvaluationTracker,
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    """
    Handle evaluation output, including logging and saving results.

    Args:
        results (Dict):
            Dictionary containing evaluation results for different tasks.
            Expected to map task names to their respective metric dictionaries.
        args (argparse.Namespace):
            Command line arguments containing configuration settings like
            output paths and logging preferences.
        evaluation_tracker (EvaluationTracker):
            Tracker object that maintains state and history of evaluation runs,
            used for metrics aggregation and progress monitoring.
        wandb_logger (Optional[WandbLogger], optional):
            Weights & Biases logger instance for experiment tracking and
            visualization. If None, W&B logging is disabled. Defaults to None.

    Returns:
        None:
            Function handles outputs via side effects (logging, saving files)
            rather than returning values.
    """

    if args.log_samples:
        print(results.keys(), results["results"].keys())
        try:
            samples = results.pop("samples")
        except KeyError:
            eval_logger.warning("log_samples is True but no samples found in results.")
            samples = {}

    dumped = json.dumps(
        results,
        indent=2,
        default=handle_non_serializable_extended,
        ensure_ascii=False,
    )
    if args.show_config:
        print(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

    if wandb_logger:
        try:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if args.log_samples:
                wandb_logger.log_eval_samples(samples)
        except Exception as e:
            eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

    evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None)
    if args.use_database and not args.debug:
        evaluation_tracker.update_evalresults_db(
            results,
            model_id=args.model_id,
            model_source=args.model,
            model_name=args.model_name,
            creation_location=args.creation_location,
            created_by=args.created_by,
            is_external=args.is_external_model,
        )

    if args.log_samples:
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

    eval_logger.info(
        f"Eval arugments: {args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), "
        f"limit: {args.limit}, num_fewshot: {args.num_fewshot}, annotator_model: {args.annotator_model}, "
        f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )

    if wandb_logger:
        wandb_logger.run.finish()


if __name__ == "__main__":
    cli_evaluate()
