import sys
import time

from PIL import Image, ImageDraw

from darknet import Darknet
from models.tiny_yolo import TinyYoloNet
from utils import *


def detect(cfgfile, weightfile, imgfile):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print("Loading weights from %s... Done!" % (weightfile))

    if model.num_classes == 20:
        namesfile = "data/voc.names"
    elif model.num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/names"

    use_cuda = 1
    if use_cuda:
        model.cuda()

    img = Image.open(imgfile).convert("RGB")
    sized = img.resize((model.width, model.height))

    for i in range(2):
        start = time.time()
        boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, "predictions.jpg", class_names)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print("Loading weights from %s... Done!" % (weightfile))

    if model.num_classes == 20:
        namesfile = "data/voc.names"
    elif model.num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/names"

    use_cuda = 1
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename="predictions.jpg",
                   class_names=class_names)


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print("Loading weights from %s... Done!" % (weightfile))

    if model.num_classes == 20:
        namesfile = "data/voc.names"
    elif model.num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/names"

    use_cuda = 1
    if use_cuda:
        model.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (model.width, model.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename="predictions.jpg",
                   class_names=class_names)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print("Usage: ")
        print("  python detect.py cfgfile weightfile imgfile")
        #detect("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights", "data/person.jpg", version=1)
