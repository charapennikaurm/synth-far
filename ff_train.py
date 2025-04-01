import click
from synth_far.training.ff_trainer import FFTrainer, FFTrainingConfig

@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the training configuration file.",
)
def train(config: str):
    config = FFTrainingConfig.from_file(config)
    trainer = FFTrainer(config)
    trainer.run()

if __name__ == "__main__":
    train()