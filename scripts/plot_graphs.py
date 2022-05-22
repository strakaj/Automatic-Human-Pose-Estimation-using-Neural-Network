from visualization import visualization_approaches, visualization_augmentations, visualization_challanges, visualization_oks, visualization_predictions

if __name__ == "__main__":
    img_format = "png"
    visualization_approaches.main(img_format)
    print("approaches visualization - finished")
    visualization_augmentations.main(img_format)
    print("augmentations visualization - finished")
    visualization_challanges.main(img_format)
    print("challanges visualization - finished")
    visualization_oks.main(img_format)
    print("oks visualization - finished")
    visualization_predictions.main(img_format)
    print("predictions visualization - finished")

