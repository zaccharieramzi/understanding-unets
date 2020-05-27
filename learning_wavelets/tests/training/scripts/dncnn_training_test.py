from learning_wavelets.training.scripts.dncnn_training import train_dncnn

def test_train_dncnn():
    train_dncnn(
        n_samples=10,
        filters=4,
        depth=4,
        n_epochs=1,
        batch_size=2,
        steps_per_epoch=2,
    )
