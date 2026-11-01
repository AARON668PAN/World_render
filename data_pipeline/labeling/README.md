# Which object is the robot handling? (G1 + Inspire, one box per sheet)

Each sheet `sheets/<clip>.jpg` shows 6 frames (2 rows of 3) or 9 frames (3 rows of 3) from the head camera of a
Unitree G1 humanoid with two five-finger Inspire hands (black fingers, white back plate) doing a household task. Each
frame is 448 x 336 pixels, drawn with a 2-pixel black border; its segment frame index is in its top-right corner (`f37`
= frame 37). Yellow ticks along the top and left edge of EACH frame mark pixel coordinates every 50 px **within that
frame**. The frames are in time order; 9-frame sheets run on past the training window so the grasp is visible.

The task is the start of the clip name, e.g. `Put_Drinks_Into_Fridge_ep00074_s0001` = putting drinks into a fridge.

## What to box: exactly ONE object

The object the robot's hand grasps, picks up, places, pushes or otherwise moves in these frames — the item the task
is about:
- drink bottle / can, vegetable, fruit, plate, cup, folder, towel, cloth, item of clothing, slipper, cushion,
  small box or part, chips can, jar;
- **a container the robot only carries is NOT the target** (e.g. the blue shopping basket held in one hand while the
  other hand picks drinks: box the drink). Box a basket / box only when lifting or moving that container IS the task
  (e.g. `Pick_Up_Basket_...` while the basket is being picked up) or it is the only thing handled;
- a wheelchair the robot pushes counts (Push_Wheelchair);
- never: the place things go into (fridge, shelf, cabinet, dishwasher, washing machine, rack, sink), furniture,
  doors, handles, the robot itself, people;
- several items handled: the one held in the most frames; tie -> the most visible one.

Pick the frame where that object is most visible (best while held, least covered by the fingers) and draw a tight box
(about 3-5 px margin) around the whole visible object in that frame's own pixel coordinates (x right, y down, origin =
that frame's top-left inside its border). Read the coordinates off the yellow ticks of that same frame.

**About to grasp counts** (10-29): if the hand reaches toward an object and grasps / pushes / moves it in a later
frame of the sheet, box that object (in any frame where it is clearly visible, e.g. before the hand covers it). The
object is drawn in every frame at inference, before the grasp too.

`none` only if the robot handles no object and does not reach for one in any frame (hands idle or out of view,
walking), or if it is impossible to tell which one of several identical objects it takes (e.g. the hand disappears
into a row of bottles and never comes back out with one), or the only object moved is moved by a person.

## Output

One JSON object per sheet, one per line, in `results/<batch>.jsonl` (batch = the bNNN of the list you were given):

    {"clip": "<clip>", "frame": 37, "box": [x0, y0, x1, y1], "name": "green bottle"}
    {"clip": "<clip>", "none": true, "why": "hands idle"}

`frame` = the f-number printed on the chosen frame; `box` integers with 0 <= x0 < x1 <= 448, 0 <= y0 < y1 <= 336.
Open every sheet yourself with Read (one Read per sheet is enough); never guess from the clip name. If an image does
not show up, write `{"clip": "<clip>", "none": true, "why": "image not visible"}` and say so in your reply.
Write the whole file in one go with Write once all sheets of the batch are done (exactly one line per clip of the list).
