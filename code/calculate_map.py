import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --- added by F
import json
import os
import re
from collections import OrderedDict

# --------------------------------------------------------------------------- #
# Helper ­functions added for the new workflow                                #
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.join(os.path.dirname(__file__), "input_output_map_v3")
GT_JSON              = os.path.join(BASE_DIR, "annotations_len_50_GT.json")
# PRED_JSON            = os.path.join(BASE_DIR, "phrase_grounded_reports_X_Ray_unproc.json")
PRED_JSON            = os.path.join(BASE_DIR, "phrase_grounded_reports_X_Ray_unproc_EDIT_AssertError_single_rerun.json")
PRED_JSON_SORTED     = os.path.join(BASE_DIR, "phrase_grounded_reports_X_Ray_unproc_sorted.json")
RAW_IO_DUMP          = os.path.join(BASE_DIR, "raw_boxes_and_classes.txt")
ABS_IO_DUMP          = os.path.join(BASE_DIR, "class_numbers_absolute_coordinates.txt")
CLEAN_IO_DUMP = os.path.join(BASE_DIR, "class_numbers_absolute_coordinates_no_empty.txt")


IMG_SIDE             = 1023  # 0-indexed image side length (1024×1024 image)

def canonical(name: str) -> str:
    """
    Normalise class names so that e.g. 'Aortic enlargement.' == 'Aortic Enlargement'
    """
    if name is None:
        return ""
    return re.sub(r"[.\s]+$", "", name.strip()).lower()

def sort_prediction_json(pred_path: str, order: list[str], out_path: str) -> dict:
    """
    Re-orders the prediction file so its keys follow the GT key order.
    Writes the sorted JSON to `out_path` and returns the loaded dict.
    """
    with open(pred_path, "r", encoding="utf8") as f:
        preds_raw = json.load(f)

    ordered = OrderedDict()
    for k in order:                       # GT order first
        if k in preds_raw:
            ordered[k] = preds_raw[k]
    for k in preds_raw:                   # any extra prediction keys afterwards
        if k not in ordered:
            ordered[k] = preds_raw[k]

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(ordered, f, indent=2)

    return ordered

def extract_boxes(gt: dict, preds: dict):
    """
    Produces four *nested* lists – one sub-list per image – suitable for the
    image-by-image COCO-style update().
    """
    pred_boxes, pred_classes = [], []
    true_boxes, true_classes = [], []

    for img_id in gt.keys():                         # guaranteed GT order
        # ---------- ground-truth ----------
        gt_boxes_img, gt_cls_img = [], []
        for x1, y1, x2, y2, cls in gt[img_id]["bbox_2d"]:
            gt_boxes_img.append([x1, y1, x2, y2])
            gt_cls_img.append(canonical(cls))
        true_boxes.append(gt_boxes_img)
        true_classes.append(gt_cls_img)

        # ---------- predictions ----------
        pb, pc = [], []
        if img_id in preds:
            for cls, entries in preds[img_id].items():
                for entry in entries:
                    # skip non-localised phrases (entry[0] is None)
                    if entry and entry[0] is not None:
                        x1, y1, x2, y2 = entry[:4]
                        pb.append([x1 * IMG_SIDE,
                                   y1 * IMG_SIDE,
                                   x2 * IMG_SIDE,
                                   y2 * IMG_SIDE])
                        pc.append(canonical(cls))
        pred_boxes.append(pb)
        pred_classes.append(pc)

    return pred_boxes, pred_classes, true_boxes, true_classes

def remove_empty_images(pb, pc, tb, tc):
    """Filter out images where *all* four entries are empty lists."""
    f_pb, f_pc, f_tb, f_tc = [], [], [], []
    for _pb, _pc, _tb, _tc in zip(pb, pc, tb, tc):
        if _pb or _pc or _tb or _tc:        # keep if any list is non‑empty
            f_pb.append(_pb)
            f_pc.append(_pc)
            f_tb.append(_tb)
            f_tc.append(_tc)
    return f_pb, f_pc, f_tb, f_tc

def build_class_mapping(true_cls: list[list[str]], pred_cls: list[list[str]]):
    """
    Builds {canonical class name -> integer ID} while guaranteeing one-to-one mapping.
    """
    classes = set()
    for lst in true_cls + pred_cls:
        classes.update(lst)
    mapping = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    return mapping

def convert_classes_to_int(cls_nested: list[list[str]], mapping: dict):
    "Replaces each string with its numeric ID (same nesting retained)."
    return [[mapping[c] for c in cls_per_img] for cls_per_img in cls_nested]




# replaces the old compute_map_supervision to support calculation for several images and prevent cross-match boxes from different images (that's what the old version did)
def compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes):
    """
    Image-by-image update so Supervision's COCO evaluator keeps images separate.
    """
    metric = MeanAveragePrecision()
    for pb, pc, tb, tc in zip(pred_boxes, pred_classes, true_boxes, true_classes):
        preds   = boxes_to_detections(pb, pc)
        targets = boxes_to_detections(tb, tc)
        metric.update(preds, targets)

    result = metric.compute()
    print("mAP@50-95:", result.map50_95)
    print("mAP@50:   ", result.map50)
    print("mAP@75:   ", result.map75)
    return result



def boxes_to_detections(boxes, class_ids=None):
    """
    Convert list of [x1, y1, x2, y2] boxes into a sv.Detections object.
    
    Args:
        boxes: List of boxes [[x1,y1,x2,y2], ...]
        class_ids: Optional list/array of class IDs for each box. 
                   If None, defaults to class_id=0 for all boxes.
    """
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    xyxy = np.array(boxes, dtype=np.float32)
    
    if class_ids is None:
        class_id_arr = np.zeros(len(boxes), dtype=int)
    else:
        class_id_arr = np.array(class_ids, dtype=int)
        if len(class_id_arr) != len(boxes):
            raise ValueError("Length of class_ids must match number of boxes")
    
    confidence = np.ones(len(boxes), dtype=np.float32)  # fixed confidence = 1.0
    
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id_arr,
        confidence=confidence
    )








def draw_boxes(pred_boxes, true_boxes, image_size=(1024, 1024)):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Draw ground truth boxes in green
    for box in true_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)  # Flip y-axis (top-left origin)
    ax.axis('off')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    re_run_on_model_output = False

    if re_run_on_model_output:
        print("---- Results based on just processed model outputs (takes .json file as input) ----")

        # --------------- STEP 1 – load GT & determine order -------------------
        with open(GT_JSON, "r", encoding="utf8") as f:
            gt_data = json.load(f, object_pairs_hook=OrderedDict)  # keep file order
        gt_order = list(gt_data.keys())

        # --------------- STEP 2 – sort predictions, write sorted file ---------
        preds_sorted = sort_prediction_json(PRED_JSON, gt_order, PRED_JSON_SORTED)

        # --------------- STEP 3 – extract raw boxes & classes -----------------
        pred_boxes, pred_classes, true_boxes, true_classes = extract_boxes(
            gt_data, preds_sorted
        )

        # dump raw (string-class, absolute-GT, *relative*-prediction) data
        with open(RAW_IO_DUMP, "w", encoding="utf8") as f:
            json.dump({
                "pred_boxes"  : pred_boxes,
                "pred_classes": pred_classes,
                "true_boxes"  : true_boxes,
                "true_classes": true_classes
            }, f, indent=2)

        # --------------- STEP 4 – build mapping & numeric conversion ----------
        class_mapping = build_class_mapping(true_classes, pred_classes)
        pred_classes_num  = convert_classes_to_int(pred_classes, class_mapping)
        true_classes_num  = convert_classes_to_int(true_classes, class_mapping)

        # --------------- STEP 5 – save absolute-coord + numeric classes -------
        with open(ABS_IO_DUMP, "w", encoding="utf8") as f:
            json.dump({
                "class_mapping": class_mapping,              # {name: id}
                "pred_boxes"   : pred_boxes,                 # already absolute
                "pred_classes" : pred_classes_num,
                "true_boxes"   : true_boxes,                 # already absolute
                "true_classes" : true_classes_num
            }, f, indent=2)


    # ---------- STEP 5b – save a "clean" version with no empty image slots ------
        clean_pb, clean_pc, clean_tb, clean_tc = remove_empty_images(
            pred_boxes, pred_classes_num, true_boxes, true_classes_num
        )
        with open(CLEAN_IO_DUMP, "w", encoding="utf8") as f:
            f.write("pred_boxes = "  + json.dumps(clean_pb, indent=4) + "\n\n")
            f.write("pred_classes = " + json.dumps(clean_pc, indent=4) + "\n\n")
            f.write("true_boxes = "  + json.dumps(clean_tb, indent=4) + "\n\n")
            f.write("true_classes = " + json.dumps(clean_tc, indent=4) + "\n")

            # --------------- STEP 6 – (optional) compute mAP ----------------------
            # Remove the next two lines if you only need the files
            compute_map_supervision(pred_boxes, pred_classes_num,
                                    true_boxes, true_classes_num)
        


    print("---- Results based on harcoded text input (same result, but does not require running the model to get the outputs)----")


    pred_boxes = [
    [ # 1 8004676ecf95af8cee446cbcd139a938
        [
            404.08500000000004,
            281.32500000000005,
            608.685,
            598.4549999999999
        ],
        [
            342.70500000000004,
            414.31500000000005,
            762.135,
            680.2950000000001
        ],
        [
            189.255,
            209.71499999999997,
            455.235,
            618.915
        ],
        [
            537.075,
            219.945,
            792.825,
            618.915
        ]
    ],
    [ # 2 e4e32ce0e061d700c0afda13faa45b1d
        [
            158.565,
            567.7650000000001,
            352.93499999999995,
            762.135
        ],
        [
            158.565,
            250.635,
            281.32500000000005,
            455.235
        ],
        [
            577.9949999999999,
            168.79500000000002,
            741.675,
            373.395
        ],
        [
            240.40499999999997,
            352.93499999999995,
            485.92499999999995,
            598.4549999999999
        ]
    ],
    [ # 3 5562ea946b0ed8574dd20d05a001d6c4
        [
            168.79500000000002,
            475.69500000000005,
            434.775,
            721.2149999999999
        ],
        [
            168.79500000000002,
            465.46500000000003,
            424.54499999999996,
            721.2149999999999
        ],
        [
            506.385,
            271.095,
            608.685,
            383.625
        ]
    ],
    [ # 4 07c12d0f562f17579aabc18c11e2ad54
        [
            455.235,
            209.71499999999997,
            629.145,
            475.69500000000005
        ],
        [
            219.945,
            97.185,
            843.9749999999999,
            598.4549999999999
        ],
        [
            393.855,
            352.93499999999995,
            792.825,
            639.375
        ],
        [
            199.485,
            158.565,
            843.9749999999999,
            629.145
        ]
    ],
    [ # 5 4a24da485b9550c8df8b19caff945cdc
        [
            393.855,
            475.69500000000005,
            823.5150000000001,
            772.365
        ],
        [
            475.69500000000005,
            352.93499999999995,
            659.835,
            629.145
        ],
        [
            506.385,
            342.70500000000004,
            649.605,
            567.7650000000001
        ]
    ],
    [ # 6 277b457e1e341a9194249937b68cd2c2
        [
            158.565,
            301.78499999999997,
            424.54499999999996,
            506.385
        ],
        [
            138.10500000000002,
            455.235,
            404.08500000000004,
            557.5350000000001
        ]
    ],
    [ # 7 af4c1f381399cfac17a6e0b983261a4e
        [
            373.395,
            445.005,
            833.7449999999999,
            792.825
        ],
        [
            455.235,
            281.32500000000005,
            649.605,
            680.2950000000001
        ]
    ],
    [ # 8 f5eb3e7e9ee9c4d08377de30251a94e2
        [
            158.565,
            117.64500000000001,
            496.155,
            751.905
        ],
        [
            618.915,
            97.185,
            915.585,
            751.905
        ],
        [
            117.64500000000001,
            567.7650000000001,
            414.31500000000005,
            762.135
        ],
        [
            168.79500000000002,
            127.875,
            496.155,
            680.2950000000001
        ],
        [
            475.69500000000005,
            250.635,
            680.2950000000001,
            608.685
        ],
        [
            618.915,
            107.41499999999999,
            843.9749999999999,
            373.395
        ],
        [
            179.02499999999998,
            342.70500000000004,
            485.92499999999995,
            629.145
        ]
    ],
    [ # 9 8de556d9cd8d026b8eba03870cc6acba
        [
            117.64500000000001,
            567.7650000000001,
            404.08500000000004,
            803.0550000000001
        ],
        [
            158.565,
            179.02499999999998,
            434.775,
            751.905
        ],
        [
            537.075,
            158.565,
            864.435,
            803.0550000000001
        ]
    ],
    [ # 10 23b0639cd035140def992b0ee7fc34f2
        [
            393.855,
            537.075,
            854.2049999999999,
            884.895
        ],
        [
            496.155,
            352.93499999999995,
            690.5250000000001,
            629.145
        ]
    ],
    [ #11 a537060564b5e08c80f46362deb565e8
        [
            158.565,
            117.64500000000001,
            434.775,
            741.675
        ],
        [
            516.615,
            107.41499999999999,
            833.7449999999999,
            823.5150000000001
        ],
        [
            158.565,
            567.7650000000001,
            352.93499999999995,
            762.135
        ],
        [
            168.79500000000002,
            127.875,
            434.775,
            710.9849999999999
        ],
        [
            158.565,
            138.10500000000002,
            434.775,
            741.675
        ],
        [
            537.075,
            138.10500000000002,
            833.7449999999999,
            813.2850000000001
        ],
        [
            158.565,
            117.64500000000001,
            424.54499999999996,
            393.855
        ],
        [
            567.7650000000001,
            168.79500000000002,
            741.675,
            373.395
        ],
        [
            179.02499999999998,
            138.10500000000002,
            424.54499999999996,
            445.005
        ],
        [
            158.565,
            250.635,
            281.32500000000005,
            455.235
        ]
    ],
    [ # 12 985be77c13eb905ee8e19a45e46ab785
        [
            5.115,
            680.2950000000001,
            342.70500000000004,
            854.2049999999999
        ],
        [
            312.015,
            424.54499999999996,
            895.125,
            803.0550000000001
        ],
        [
            5.115,
            66.495,
            1017.885,
            1017.885
        ]
    ]
    ]

    pred_classes = [
    [
        0,
        2,
        14,
        14
    ],
    [
        11,
        15,
        10,
        6
    ],
    [
        8,
        3,
        1
    ],
    [ # 4
        0,
        5,
        2,
        12
    ],
    [
        2,
        0,
        1
    ],
    [
        8,
        11
    ],
    [
        2,
        0
    ],
    [
        14,
        14,
        11,
        12,
        0,
        10,
        8
    ],
    [
        11,
        14,
        14
    ],
    [
        2,
        0
    ],
    [
        4,
        4,
        11,
        12,
        14,
        14,
        13,
        10,
        7,
        15
    ],
    [
        11,
        2,
        10
    ]
    ]

    true_boxes = [
    [
        [
            500.8430480957031,
            290.5226745605469,
            598.9346313476562,
            401.96856689453125
        ],
        [
            372.693115234375,
            501.3167419433594,
            675.936279296875,
            596.539306640625
        ],
        [
            694.1629638671875,
            502.6311340332031,
            773.9834594726562,
            551.4470825195312
        ],
        [
            258.2164306640625,
            471.60888671875,
            268.00213623046875,
            481.10577392578125
        ]
    ],
    [
        [
            162.7555389404297,
            739.274658203125,
            217.0937957763672,
            803.2019653320312
        ],
        [
            680.5592651367188,
            311.6768493652344,
            766.9830932617188,
            408.3891906738281
        ],
        [
            734.8471069335938,
            418.0252380371094,
            769.1422729492188,
            450.2626647949219
        ],
        [
            347.874755859375,
            141.3868865966797,
            479.515380859375,
            253.2342071533203
        ]
    ],
    [
        [
            184.46543884277344,
            594.2944946289062,
            293.9931335449219,
            702.0465087890625
        ],
        [
            153.7237548828125,
            581.1856079101562,
            323.58843994140625,
            736.0020141601562
        ],
        [
            740.7280883789062,
            273.5866394042969,
            758.1695556640625,
            295.6255798339844
        ],
        [
            830.2438354492188,
            200.78244018554688,
            913.9707641601562,
            291.34478759765625
        ],
        [
            742.8372192382812,
            273.7303771972656,
            757.7689819335938,
            296.1264343261719
        ],
        [
            739.210693359375,
            275.3974914550781,
            760.491455078125,
            295.6870422363281
        ]
    ],
    [
        [
            529.3150024414062,
            240.260498046875,
            611.6900024414062,
            313.3806457519531
        ],
        [
            230.23365783691406,
            262.4533386230469,
            419.8033447265625,
            492.2333068847656
        ],
        [
            618.7533569335938,
            364.0066833496094,
            762.61669921875,
            508.42333984375
        ],
        [
            666.7733154296875,
            443.183349609375,
            803.8866577148438,
            508.6233215332031
        ],
        [
            379.5899963378906,
            434.7900085449219,
            727.5432739257812,
            533.2666625976562
        ],
        [
            309.94732666015625,
            115.80899810791016,
            433.9766845703125,
            139.50433349609375
        ]
    ],
    [
        [
            455.1134338378906,
            558.7681274414062,
            829.96630859375,
            736.0115966796875
        ],
        [
            221.86843872070312,
            528.1942138671875,
            480.91650390625,
            727.0809936523438
        ],
        [
            541.0288696289062,
            355.4809875488281,
            635.9878540039062,
            448.73114013671875
        ],
        [
            545.0477294921875,
            356.35693359375,
            635.4081420898438,
            445.80865478515625
        ]
    ],
    [
        [
            215.3477783203125,
            230.55714416503906,
            394.4284973144531,
            369.88623046875
        ],
        [
            168.4426727294922,
            261.4141540527344,
            393.8922119140625,
            454.7377624511719
        ]
    ],
    [
        [
            391.1789855957031,
            562.1729736328125,
            813.3969116210938,
            719.84033203125
        ],
        [
            533.8176879882812,
            296.8363342285156,
            648.0689697265625,
            438.6634521484375
        ]
    ],
    [
        [
            279.1871337890625,
            485.3576965332031,
            392.3382873535156,
            581.5360107421875
        ],
        [
            858.5943603515625,
            710.2216796875,
            898.0546875,
            770.9943237304688
        ],
        [
            773.7462158203125,
            596.5348510742188,
            918.7271118164062,
            799.6982421875
        ],
        [
            858.5943603515625,
            710.2216796875,
            898.0546875,
            770.9943237304688
        ],
        [
            773.7462158203125,
            596.5348510742188,
            918.7271118164062,
            799.6982421875
        ],
        [
            586.3640747070312,
            251.1140594482422,
            679.9659423828125,
            367.8728942871094
        ],
        [
            279.34490966796875,
            482.76116943359375,
            341.584716796875,
            543.7302856445312
        ],
        [
            810.7002563476562,
            646.2582397460938,
            895.98291015625,
            744.2880249023438
        ]
    ],
    [
        [
            155.36392211914062,
            605.149658203125,
            358.74853515625,
            774.6891479492188
        ],
        [
            262.8825378417969,
            341.9202575683594,
            394.57318115234375,
            542.761474609375
        ],
        [
            181.1703643798828,
            585.1573486328125,
            382.22808837890625,
            706.239990234375
        ]
    ],
    [
        [
            421.9848327636719,
            652.6238403320312,
            799.2166748046875,
            818.24560546875
        ],
        [
            568.9026489257812,
            346.3203430175781,
            668.4970703125,
            458.90631103515625
        ]
    ],
    [
        [
            147.58021545410156,
            261.8802185058594,
            413.3599853515625,
            789.4677734375
        ],
        [
            141.93099975585938,
            779.308349609375,
            220.32501220703125,
            853.4450073242188
        ],
        [
            141.93099975585938,
            779.308349609375,
            220.32501220703125,
            853.4450073242188
        ],
        [
            308.4371643066406,
            155.56150817871094,
            369.8050231933594,
            302.461181640625
        ],
        [
            221.7306671142578,
            299.33331298828125,
            416.0233154296875,
            437.6216735839844
        ],
        [
            615.7033081054688,
            136.86033630371094,
            666.5532836914062,
            284.9471740722656
        ],
        [
            137.03866577148438,
            508.5,
            356.3000183105469,
            801.239990234375
        ],
        [
            187.1890106201172,
            107.29833221435547,
            444.9366760253906,
            428.0266418457031
        ],
        [
            137.03866577148438,
            508.5,
            356.3000183105469,
            801.239990234375
        ],
        [
            401.7866516113281,
            346.3866882324219,
            526.5800170898438,
            709.1033325195312
        ],
        [
            187.1890106201172,
            107.29833221435547,
            444.9366760253906,
            428.0266418457031
        ],
        [
            554.7433471679688,
            119.3636703491211,
            758.1399536132812,
            288.8609924316406
        ],
        [
            304.9840087890625,
            118.96133422851562,
            443.7733154296875,
            157.44866943359375
        ],
        [
            601.2200317382812,
            121.29399871826172,
            679.3633422851562,
            148.1183319091797
        ],
        [
            288.0090026855469,
            477.0933532714844,
            366.5133361816406,
            642.1300048828125
        ],
        [
            187.1890106201172,
            107.29833221435547,
            444.9366760253906,
            428.0266418457031
        ],
        [
            278.0053405761719,
            327.2386779785156,
            312.1146545410156,
            383.3866882324219
        ],
        [
            600.9833374023438,
            384.67333984375,
            776.296630859375,
            788.4299926757812
        ]
    ],
    [
        [
            786.0150756835938,
            579.322998046875,
            956.83154296875,
            758.0122680664062
        ],
        [
            291.62298583984375,
            477.64984130859375,
            925.6076049804688,
            769.2301025390625
        ],
        [
            218.4761505126953,
            20.167125701904297,
            515.1280517578125,
            280.54248046875
        ],
        [
            5.877826690673828,
            127.35922241210938,
            283.35516357421875,
            490.98028564453125
        ],
        [
            456.4491882324219,
            286.41265869140625,
            640.622802734375,
            966.6895751953125
        ]
    ]
    ]

    true_classes = [
    [
        0,
        2,
        14,
        9
    ],
    [
        11,
        15,
        10,
        6
    ],
    [
        8,
        3,
        9,
        10,
        10,
        1
    ],
    [ # 4
        0,
        5,
        5,
        6,
        2,
        12
    ],
    [
        2,
        10,
        0,
        1
    ],
    [
        8,
        11
    ],
    [
        2,
        0
    ],
    [
        14,
        11,
        11,
        12,
        12,
        0,
        10,
        8
    ],
    [
        11,
        14,
        8
    ],
    [
        2,
        0
    ],
    [
        4,
        11,
        12,
        14,
        14,
        14,
        13,
        13,
        10,
        10,
        10,
        4,
        12,
        12,
        14,
        7,
        15,
        5
    ],
    [
        11,
        2,
        10,
        10,
        10
    ]
    ]




    compute_map_supervision(pred_boxes, pred_classes,
                            true_boxes, true_classes)
    
