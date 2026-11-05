# Demo answer

Computer science. Parts 1 and 3 use Prototypes #3 and #4. Part 2 compares #1 and #3 only.

## How It Works (30%)

Mr. Zhang lives alone, has fallen before, and may forget a daily task. I combine the camera fall alert with the desk companion. Neither has to be worn. Blood pressure stays with his usual care.

`Camera → pose estimation → fall if he drops and stays down → message to family`

`Clock + touch → medication prompt and animated eyes → no touch by the deadline → message to family`

The camera covers a fall nobody else will see. The companion covers a missed dose.

## Comparing Approaches (30%)

**#1 wearable.** Biggest advantage: privacy. It sends a motion alert and does not film the home. Biggest disadvantage: he must wear and charge it. With mild memory loss, a forgotten clip means a fall produces no alert.

**#3 camera.** Biggest advantage: no wearing step, so the monitored room is still covered if he forgets a device. Biggest disadvantage: a camera in a private room, plus false alarms from sitting down, bending, or a pet.

## Your Disciplinary Perspective (40%)

From computer science, the limit of Prototype #3 is generalization. The demo is one room, one camera angle, and one person. A real home changes light, furniture, and posture, so demo accuracy does not show that the same model works there. A missed fall has no witness; repeated false alerts teach the family to ignore them.

Evaluation: label a fall as down and not up within a fixed window. Split by household, not by frame. Choose the threshold on training homes only. On new homes, report sensitivity and false alarms per day, not accuracy. Use it only if sensitivity stays at least 0.90 and false alarms stay under one per day. If not, alert only after the horizontal posture persists.
