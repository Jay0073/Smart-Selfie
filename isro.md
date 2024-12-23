# navigation techniques
- no GNSS, pseudolite, reflector arrays

need to develop and demonstrate the autonomous capabilities of anav
  for navigation and guidance
  to identify safe landing spots and perform stable landing and takeoff

- stable vertical takeoff, hover and landing
- scan the arena and identify the boundary, safe spots for landing
- choose the sequence and stable landing at safe spot
- after safe landing, ANAV shall return to the home position

target spots should be based on surface topography


## ANAV features
- can also use readily available hardware platform
- mass of 2kg
- use of satellite-based navigation systems, external markers, or local
positioning systems is strictly prohibited.
- ANAV shall have an emergency call of mode, inc ase of exigencies.

## ANAV specifications
- rotor craft
    micro drone mass < 2kg
- indigenously developed softwares/algorithms
- Battery powered
- slope landing capability of minimum 15 deg
- expected height of flight 10 to 15m
- should cover 40ft in width and 30ft in height

## few definitions
- base station (laptop) controls the initial command and emergency command, these commands and are also wireless
- ANAV will deliver information of
    - position (x, y, height)
    - vertical velocity, horizontaly velocity
    - safe site co-ordinates
    - battery level
    - others if any
- ANAV modes
    - manual mode (operations from base station)
    - autonomous mode (after receiving auto-start command, tasks in rounds)
    - safe mode (during emergency conditions (low-battery, lost-link, etc..))

- emergency conditions
    - any hint of collision or malfunction of software or hardware, control system etc
    - unforeseen deviations

- home positions
- safe spot      both of approx of 1.5sq m

- qualification operations to be done
    - vertical takeoff and maintains the vertical course with constant speed
    - after reaching 3m of height, it should hover for 30sec min
    - landing at the same place 