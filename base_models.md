# Evaluate base models

To evaluate base models we need to fomat each question as a continueation since we cannnot use chat template. Curewnt template is following:
```
<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<think>
```

You can also change the system instruction by providing `--system_instruction` argument.

Example of a command:

```bash
CONFIG=configs/reasoning_base_models.yaml

cd $EVALCHEMY_HOME

python -m eval.eval \
    --model hf \
    --config $CONFIG \
    --model_args pretrained=$MODEL_DIR,trust_remote_code=True \
    --output_path $EVAL_DIR \
    --max_tokens 4096 \
    --system_instruction "You are a helpful AI assistant." \
    --apply_chat_template False \
    --n_repeat 
```