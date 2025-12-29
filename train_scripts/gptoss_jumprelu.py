import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    JumpReLUTrainingSAEConfig,
    LoggingConfig,
)
from sae_lens.load_model import load_model
from datasets import load_dataset

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

gptoss_model_name = "openai/gpt-oss-20b"
gptoss_d_in = 2880
gptoss_d_sae = gptoss_d_in * 4
gptoss_hook_name = "model.layers.23.mlp"


model = load_model(model_name=gptoss_model_name, model_class_name = "AutoModelForCausalLM", device=device, model_from_pretrained_kwargs={"device_map": "auto"})
stream_ds = load_dataset(
    "json",
    data_files=[
        "/shared/local-shard_0_of_10/shard_00000001_processed.jsonl.zst",
        # "/shared/local-shard_0_of_10/shard_00000003_processed.jsonl.zst",
    ],
    split="train",
    streaming=True,
).remove_columns(["bff_contained_ngram_count_before_dedupe", "language_id_whole_page_fasttext", "metadata", "previous_word_count", "url", "warcinfo", 
                "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob"]
).filter(lambda x: len(x['text']) < 1024)


total_training_steps = 30_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    hook_name=gptoss_hook_name,  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    # dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    # is_dataset_tokenized=True,
    # streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    sae=JumpReLUTrainingSAEConfig(
        l0_coefficient=5.0, # Sparsity penalty coefficient
        jumprelu_sparsity_loss_mode="tanh",
        jumprelu_tanh_scale=4.0, # default value
        jumprelu_bandwidth=2.0,
        jumprelu_init_threshold=0.1,
        pre_act_loss_coefficient=3e-6,
        l0_warm_up_steps=total_training_steps,
        d_in=gptoss_d_in,  # the width of the mlp output.
        d_sae=gptoss_d_sae,  # the width of the SAE. Larger will result in better stats but slower training.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_activations="expected_average_only_in",
        # Router entropy settings (optional)
        use_router_entropy=True,  # Enable router entropy adjustment
        router_entropy_layer="model.layers.23.mlp.router",  # Router layer hook name
        router_entropy_weight=1.0,  # Weight for router entropy adjustment (default: 0.1)
    ),
    # Training Parameters
    lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    train_batch_size_tokens=batch_size,
    context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="sae_lens_tutorial",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),
    # Misc
    device=device,
    seed=42,
    n_checkpoints=1,
    checkpoint_path="/root/nas/checkpoints",
    save_final_checkpoint=True,
    dtype="float32",
    autocast=True,
    autocast_lm=True,
    compile_sae=True,
    compile_llm=True,
)
# look at the next cell to see some instruction for what to do while this is running.
sparse_autoencoder = SAETrainingRunner(
    cfg,
    override_model = model,
    override_dataset = stream_ds,
).run() 