"""Kon-Tiki 1000 biochar volume from a single photo (frustum geometry from the
Ithaka design drawing). Method = the proven PoC: segment the biochar SURFACE, read
its size relative to the KNOWN rim, convert to litres via the known cone shape."""
import numpy as np

# --- Kon-Tiki 1000 geometry (from design_Kon-Tiki 1000.pdf, FRONT ELEVATION) ---
R   = 75.0     # top rim radius   (Ø1500 mm)
RB  = 41.15    # bottom radius    (Ø823 mm) -> truncated cone (frustum)
H   = 93.0     # cone depth cm    (930 mm)
DENS = 0.25    # biochar bulk density kg/L (typical 0.20-0.30)

def V_full():  # full frustum capacity
    return (1/3)*np.pi*H*(RB**2 + RB*R + R**2)

def volume_litres(u):
    """u = (biochar surface circle radius) / (rim circle radius), measured in the
    photo. Scale-FREE ratio. Returns biochar volume in litres."""
    rs = u*R
    h  = np.clip(H*(rs-RB)/(R-RB), 0, H)          # fill height from bottom
    rs = RB + (R-RB)*(h/H)
    return (1/3)*np.pi*h*(RB**2 + RB*rs + rs**2) / 1000.0   # cm3 -> L

VF = V_full()/1000.0
print(f"Kon-Tiki 1000 : rim R={R}cm  bottom r={RB}cm  depth H={H}cm")
print(f"Full capacity (geometry) = {VF:.0f} L   (drawing label: 1000 L)  -> match: {100*VF/1000:.1f}%\n")
print(f"{'surf/rim u':>11}{'fill h (cm)':>12}{'volume (L)':>11}{'fill %':>8}{'weight kg':>11}")
for u in [0.535,0.60,0.70,0.80,0.90,0.95,1.00]:
    V = volume_litres(u); rs=u*R; h=np.clip(H*(rs-RB)/(R-RB),0,H)
    print(f"{u:>11.3f}{h:>12.1f}{V:>11.1f}{100*V/VF:>7.0f}%{V*DENS:>11.1f}")
print("\nHow to read 'u' from a photo (proven PoC method):")
print(" 1. Photograph the kiln from ~50-60 deg above (angle-montage sweet spot).")
print(" 2. AI segments the biochar SURFACE circle and the kiln RIM circle.")
print(" 3. u = surface-circle radius / rim-circle radius  (ratio -> no scale marker needed).")
print(" 4. volume_litres(u) -> litres ; x density -> kg.")
