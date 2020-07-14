#!/bin/bash
wandb_key=0
source /app/secret_key.txt

wandb login ${wandb_key}
