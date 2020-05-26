from learning_wavelets.training.scripts.learnlet_subclassed_training import train_learnlet

def test_train_learnlet():
    train_learnlet(
        n_samples=10,
        n_filters=4,
        n_epochs=1,
        batch_size=2,
        steps_per_epoch=2,
    )
