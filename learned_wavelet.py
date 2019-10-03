image = Input((None, None, n_channel))

details_thresholded = Conv2D(activation='relu')(image)
coarse = Conv2D(activation='linear')(image)

reconstructed_coarse = smthg(coarse)

reconstructed_image = Conv2D(activation='linear')(concatenate([reconstructed_coarse, details_thresholded]))
