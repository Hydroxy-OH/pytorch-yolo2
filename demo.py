import cv2

from darknet import Darknet
from utils import *


def demo(cfgfile, weightfile):
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
    class_names = load_class_names(namesfile)

    use_cuda = 1
    if use_cuda:
        model.cuda()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (model.width, model.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            print("------")
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)
            cv2.waitKey(1)
        else:
            print("Unable to read image")
            exit(-1)


############################################
if __name__ == "__main__":
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
        # demo("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights")
    else:
        print("Usage:")
        print("    python demo.py cfgfile weightfile")
        print("")
        print("    perform detection on camera")
