# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import cv2
import numpy as np


# %%
def to_homogeneous(pts):
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def from_homogeneous(pts):
    return pts[:, :2] / pts[:, 2][:, None]


# %% [markdown]
# # Getting +showing matches

# %%
def get_matches(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)[:50]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return pts1, pts2, kp1, kp2, good

def draw_matches(img1, img2, kp1, kp2, matches):
    match_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save instead of showing
    cv2.imwrite("matches_output.jpg", match_img)


# %% [markdown]
# # Compute the Homography Parameters

# %%
def compute_homography(pts_src, pts_dst):
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)

    assert len(pts_src) >= 4 and len(pts_src) == len(pts_dst)

    A = []

    for (x, y), (xp, yp) in zip(pts_src, pts_dst):
        A.append([-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp])
        A.append([0, 0, 0, -x, -y, -1, yp*x, yp*y, yp])

    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]

    H = h.reshape(3, 3)

    if abs(H[2, 2]) < 1e-8:
        return np.eye(3)

    H = H / H[2, 2]

    return H
def verify_homography(H, pts_src, pts_dst, label=""):
    print(f"\n── Verification {label} ──")
    proj = from_homogeneous((H @ to_homogeneous(pts_src).T).T)
    for i in range(min(5, len(pts_src))):
        err = np.linalg.norm(proj[i] - pts_dst[i])
        print(f"  [{i}] src={pts_src[i].round(1)}  "
              f"proj={proj[i].round(1)}  "
              f"gt={pts_dst[i].round(1)}  err={err:.2f}px")


# %% [markdown]
# # Forward Wrap

# %%
# 3. Forward Warping
def forward_warp(src, H, out_shape):
    h_out, w_out = out_shape
    dst = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            pt = np.array([x, y, 1])
            mapped = H @ pt
            mapped /= mapped[2]

            x_new, y_new = int(mapped[0]), int(mapped[1])

            if 0 <= x_new < w_out and 0 <= y_new < h_out:
                dst[y_new, x_new] = src[y, x]

    return dst


# %% [markdown]
# # Inverse Warp

# %%
def inverse_warp(src, H, out_shape):
    H_inv = np.linalg.inv(H)

    h_out, w_out = out_shape
    dst = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    h_src, w_src = src.shape[:2]

    for y in range(h_out):
        for x in range(w_out):

            pt = np.array([x, y, 1])
            src_pt = H_inv @ pt
            src_pt /= src_pt[2]

            xs, ys = src_pt[0], src_pt[1]

            if 0 <= xs < w_src - 1 and 0 <= ys < h_src - 1:

                x0, y0 = int(xs), int(ys)
                x1, y1 = x0 + 1, y0 + 1

                dx = xs - x0
                dy = ys - y0

                top = (1 - dx) * src[y0, x0] + dx * src[y0, x1]
                bottom = (1 - dx) * src[y1, x0] + dx * src[y1, x1]
                dst[y, x] = (1 - dy) * top + dy * bottom

    return dst


# %% [markdown]
# # Overlay

# %%
def overlay(frame, warped, mask):
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    warped_fg = cv2.bitwise_and(warped, warped, mask=mask)

    return cv2.add(frame_bg, warped_fg)


# %% [markdown]
# # RANSAC filtering

# %%
def filter_matches_ransac(pts1, pts2):
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    if len(pts1) < 4:
        return pts1[:0], pts2[:0]

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    if mask is None:
        return pts1[:0], pts2[:0]

    mask = mask.ravel().astype(bool)

    return pts1[mask], pts2[mask]


# %% [markdown]
# # Main Pipeline

# %%
def main():
    book = cv2.imread(r"assignment_2_materials\cv_cover.jpg")
    video = cv2.VideoCapture(r"assignment_2_materials\book.mov")
    ar = cv2.VideoCapture(r"assignment_2_materials\ar_source.mov")

    if book is None or not video.isOpened() or not ar.isOpened():
        print("Error loading resources.")
        return

    h, w = book.shape[:2]

    # -----------------------------
    # FIRST FRAME INITIALIZATION
    # -----------------------------
    ret_first, first_frame = video.read()
    if not ret_first:
        print("Failed to read first frame")
        return

    pts1, pts2, kp1, kp2, good = get_matches(book, first_frame)
    pts1, pts2 = filter_matches_ransac(pts1, pts2)

    print("Number of matches (Part 1):", len(good))
    draw_matches(book, first_frame, kp1, kp2, good)

    H = compute_homography(pts1, pts2)
    verify_homography(H, pts1, pts2)

    # reset video
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # output setup
    ret_tmp, tmp_frame = video.read()
    frame_h, frame_w = tmp_frame.shape[:2]
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ar_output.mp4', fourcc, 30.0, (frame_w, frame_h))

    # -----------------------------
    # LOOP
    # -----------------------------
    frame_count = 0

    while True:
        ret, frame = video.read()
        ret2, ar_frame = ar.read()

        if not ret or not ret2:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # -----------------------------
        # UPDATE HOMOGRAPHY EVERY 10 FRAMES
        # -----------------------------
        if frame_count % 10 == 0:
            pts1, pts2, _, _, _ = get_matches(book, frame)
            pts1, pts2 = filter_matches_ransac(pts1, pts2)

            if len(pts1) >= 4:
                H = compute_homography(pts1, pts2)

        if H is None:
            continue

        # -----------------------------
        # AR FRAME PREPROCESS
        # -----------------------------
        h_ar, w_ar = ar_frame.shape[:2]
        aspect_book = w / h
        aspect_ar = w_ar / h_ar

        if aspect_ar > aspect_book:
            new_w = int(h_ar * aspect_book)
            start_x = (w_ar - new_w) // 2
            ar_frame = ar_frame[:, start_x:start_x + new_w]
        else:
            new_h = int(w_ar / aspect_book)
            start_y = (h_ar - new_h) // 2
            ar_frame = ar_frame[start_y:start_y + new_h, :]

        ar_frame = cv2.resize(ar_frame, (w, h))

        # -----------------------------
        # WARP + OVERLAY
        # -----------------------------
        warped = inverse_warp(ar_frame, H, (frame.shape[0], frame.shape[1]))

        corners = np.float32([[0,0],[w,0],[w,h],[0,h]])
        mapped = []

        for c in corners:
            pt = np.array([c[0], c[1], 1])
            dst = H @ pt
            dst /= dst[2]
            mapped.append([dst[0], dst[1]])

        mapped = np.array(mapped, dtype=np.int32)

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, mapped, 255)

        result = overlay(frame, warped, mask)

        out.write(result)

    video.release()
    ar.release()
    out.release()
    print("Video saved as ar_output.mp4")

if __name__ == "__main__":    main()
