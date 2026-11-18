"""Pre/post-processors of "dit_resnet": those of multi_task_dit (renaming, batch dim, CLIP tokenizer for the task text,
device, normalisation per the config's mapping)."""
from lerobot.policies.multi_task_dit.processor_multi_task_dit import make_multi_task_dit_pre_post_processors


def make_dit_resnet_pre_post_processors(config, dataset_stats=None):
    return make_multi_task_dit_pre_post_processors(config, dataset_stats)
