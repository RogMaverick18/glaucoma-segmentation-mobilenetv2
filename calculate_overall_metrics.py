"""
Simple calculator for overall Dice and IoU metrics
Paste your per-class results when prompted
"""

import re

print("="*60)
print("OVERALL DICE & IoU CALCULATOR")
print("="*60)
print("\nPaste your per-class metrics (3 lines), then press Enter twice:")
print("\n" + "-"*60)

# Read multi-line input
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)

# Parse the metrics
dice_values = []
iou_values = []

for line in lines:
    # Extract Dice and IoU values using regex (supports both formats)
    # Format 1: Dice=0.9993, IoU=0.9986
    # Format 2: Dice: 0.9993, IoU: 0.9986
    dice_match = re.search(r'Dice[=:]?\s*(\d+\.\d+)', line)
    iou_match = re.search(r'IoU[=:]?\s*(\d+\.\d+)', line)
    
    if dice_match:
        dice_values.append(float(dice_match.group(1)))
    if iou_match:
        iou_values.append(float(iou_match.group(1)))

# Calculate overall metrics
if len(dice_values) == 3 and len(iou_values) == 3:
    overall_dice = sum(dice_values) / 3
    overall_iou = sum(iou_values) / 3
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Overall Dice: {overall_dice:.4f}")
    print(f"Overall IoU:  {overall_iou:.4f}")
    print("="*60)
else:
    print("\n✗ Error: Could not parse 3 classes. Please check your input format.")
