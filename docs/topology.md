# Topology & State Transitions

## Overview
If the **State Space** defines *where* a system can exist, the **Topology** defines *how* it moves. 

A topology is defined strictly with respect to a system. At its most basic level, it establishes the physical or logical connections between states, dictating the valid transitions a system can make in a single tick. By defining these local, single-step rules, the topology organically generates the complex, global bounds of the simulation.

## Single-Step Transitions: Neighbors and Inverses

Given a system currently occupying a state $s$:

* **Reachable Set / Neighbors ($V[s]$):** The set of all possible states the system can transition to in exactly one forward step. 
* **Reaching Set / Inverse Neighbors ($R[s]$):** The inverse of $V[s]$. It defines the set of all previous states that could have legally transitioned *into* $s$ in a single step.

## Multi-Step Dynamics: Frontiers and Basins

In a single step, the system can only move to $V[s]$. However, over multiple ticks ($l$ steps), the system's potential positions expand into a wave. We define this expansion mathematically using Frontiers (the leading edge) and Basins (the total covered area).

### 1. Forward Expansion (Where can the system go?)

* **Multi-Step Forward Frontier ($F^l[s]$):** The set of all states the system could occupy in *exactly* $l$ steps. It is the leading edge of the probability wave.
$$F^0[s]=\{s\}$$
$$F^l[s]=\bigcup_{x\in F^{l-1}[s]}V[x]$$

* **Multi-Step Forward Basin ($B^l[s]$):** The set of all states the system has *potentially been in* up to $l$ steps. It encompasses the starting state and all expanding frontiers.
$$B^l[s]=\bigcup_{k=0}^l F^k[s]$$

### 2. Backward Expansion (Where did the system come from?)

By running the topology in reverse using the Reaching Set ($R[s]$), we can trace the historical causality of a specific state. 

* **Multi-Step Backward Frontier ($F^{-l}[s]$):** The exact set of states from which the system must have originated exactly $l$ steps ago to arrive at $s$.
$$F^{-0}[s]=\{s\}$$
$$F^{-l}[s]=\bigcup_{x\in F^{-(l-1)}[s]}R[x]$$

* **Multi-Step Backward Basin ($B^{-l}[s]$):** The complete set of all historical states that could have possibly led to $s$ within a window of $l$ steps.
$$B^{-l}[s]=\bigcup_{k=0}^l F^{-k}[s]$$

---

## Example: The 2D Grid

Imagine a particle on a 2D infinite grid, starting at state $s=(0,0)$. The topology allows movement Up, Down, Left, or Right.

* **$V[s]$ (1-Step):** The particle can move to $(0,1),(0,-1),(-1,0),(1,0)$.
* **$F^3[s]$ (3-Step Frontier):** The particle forms a diamond-shaped perimeter exactly 3 units away from the origin. It cannot be at $(0,0)$ because 3 steps forces it outward.
* **$B^3[s]$ (3-Step Basin):** The solid, filled-in diamond of every coordinate the particle could have touched over the course of those 3 steps, including backtracking to the origin.

> 🖼️ **Visualizing the Wave:**
> 
> *[Placeholder: Insert animated GIF here showing a single point recursively expanding its $F^l[s]$ outward tick-by-tick, turning into a hollow diamond/circle, alongside a second GIF showing the $B^l[s]$ filling in as a solid shape.]*