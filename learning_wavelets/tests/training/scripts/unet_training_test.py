from learning_wavelets.training.scripts.unet_training import train_unet

def test_train_unet():
    train_unet(
        n_samples=10,
        base_n_filters=4,
        n_epochs=1,
        batch_size=2,
        steps_per_epoch=2,
    )
