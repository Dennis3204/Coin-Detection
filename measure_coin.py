#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import math

def filter_overlaps(objects, tol=0.3):
    """
    Cluster circles whose centers lie within tol * radius of a larger circle
    and keep only the largest circle in each cluster.
    """
    # sort descending by diameter
    objs = sorted(objects, key=lambda o: o['diameter_px'], reverse=True)
    filtered = []

    for o in objs:

        cx, cy = o['center']

        # if this circle lies within tol*radius of any kept circle, skip it
        if any(
            math.hypot(cx - k['center'][0], cy - k['center'][1])
            < k['diameter_px'] * tol
            for k in filtered
        ):
            
            continue
        filtered.append(o)

    return filtered


def process_image(path, scale):
    
    img = cv2.imread(path)
    if img is None:
        return None, []

    # --- resize to 800px width ---
    h, w = img.shape[:2]
    if w > 800:
        f = 800.0 / w
        img = cv2.resize(img, None, fx=f, fy=f)

    # --- classical segmentation ---
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- build circle list ---
    objects = []
    oid = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        diameter_px = 2 * r
        diameter_mm = diameter_px * scale if scale else None
        objects.append({
            'id'         : oid,
            'center'     : (int(x), int(y)),
            'diameter_px': diameter_px,
            'diameter_mm': diameter_mm
        })
        oid += 1

    # --- filter out overlapping duplicates ---
    objects = filter_overlaps(objects, tol=0.3)

    return img, objects

def main():

    p = argparse.ArgumentParser()

    p.add_argument('--input-dir', default='test_IMG',
                   help='folder with input images')
    p.add_argument('--scale', type=float, default=None,
                   help='mm per pixel (optional)')
    
    args = p.parse_args()

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    quit_flag = False
    current_objects = []
    base_img = None

    def on_click(evt, x, y, flags, param):

        nonlocal base_img, current_objects

        if evt != cv2.EVENT_LBUTTONDOWN or not current_objects:
            return
        
        # find nearest circle
        dists = [
            (math.hypot(x - o['center'][0], y - o['center'][1]), o)
            for o in current_objects
        ]

        d, o = min(dists, key=lambda t: t[0])

        if d > o['diameter_px'] / 2:
            return
        
        msg = f"Object {o['id']}: {o['diameter_px']:.1f} px"

        if o['diameter_mm'] is not None:
            msg += f", {o['diameter_mm']:.2f} mm"

        print(msg)

        disp = base_img.copy()

        cv2.circle(disp, o['center'], int(o['diameter_px']/2), (0, 0, 255), 2)
        cv2.imshow('image', disp)

    cv2.setMouseCallback('image', on_click)

    for fname in sorted(os.listdir(args.input_dir)):
        
        path = os.path.join(args.input_dir, fname)
        if not os.path.isfile(path):
            continue

        img, objs = process_image(path, args.scale)
        if img is None:
            continue

        base_img = img.copy()
        current_objects = objs

        # annotate final circles
        for o in objs:
            cv2.circle(base_img, o['center'], int(o['diameter_px']/2), (0, 255, 0), 1)
            cv2.putText(
                base_img,
                str(o['id']),
                (o['center'][0] + 5, o['center'][1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

        cv2.imshow('image', base_img)
        print(f"\n{fname}: detected {len(objs)} objects.")
        print("Click an object to see its size. Press 'n' for next, 'q' to quit.")

        # navigation
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                break
            if key == ord('q'):
                quit_flag = True
                break
        if quit_flag:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
