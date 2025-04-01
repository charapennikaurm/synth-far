import click
from synth_far.training.age_trainer import AGETrainer, AGETrainingConfig

@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the training configuration file.",
)
def train(config: str):
    config = AGETrainingConfig.from_file(config)
    trainer = AGETrainer(config)
    trainer.run()

if __name__ == "__main__":
    train()