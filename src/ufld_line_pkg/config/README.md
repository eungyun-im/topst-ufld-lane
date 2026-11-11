# ufld_line_pkg Config Notes

This package is intended to replace `line_pkg` at the lane-detection boundary:

```text
subscribe: /camera/image/compressed
publish:   /line_offset
publish:   /yellow_detected
```

The control package can keep subscribing to `/line_offset`.

The launch defaults assume the D-Racer camera is configured as:

```text
IMAGE_WIDTH: 320
IMAGE_HEIGHT: 160
```

The UFLD head-shape parameters must match the exported model. If a custom
320x160 UFLD v2 model uses different anchors or grid counts, update
`launch/ufld_line.launch`.
