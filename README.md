## What is this?

This code might be useful to you if you have recorded two speakers (with their own microphones) talking to each other in the same room. Depending on the speakers' relative positioning, the room acoustics and microphone placements, the microphones might have picked up a significant amount of crosstalk (or "mic bleed").  In such a case a simple noise gate applied independently on each channel might not work because it is difficult to distinguish between wanted speech and unwanted cross talk.

This code implements a "dynamic noise gate" effect which analyses both microphone recordings at the same time to derive dynamic noise gate thresholds for both signals taking crosstalk into account. It assumes that the short-term RMS values of the speach signals, noise floors and recorded signals are easily predictable as follows:

```math
\begin{pmatrix}
C_a^2 \\ C_b^2
\end{pmatrix}
= \begin{bmatrix}
    1 & \alpha^2 \\
    \beta^2 & 1
\end{bmatrix}
\begin{pmatrix}
    S_a^2 \\ S_b^2
\end{pmatrix}
+ \begin{pmatrix}
    N_a^2 \\ N_b^2
\end{pmatrix}
```

Here, $C_a$ and $C_b$ are the RMS values for recorded blocks of two channels (a and b), $\alpha$ and $\beta$ refer to gain factors for the crosstalk, $S_a$ and $S_b$ are the "clean" speach signals and $N_a$ and $N_b$ are recording noise levels that covers everything but the crosstalk. We assume $\alpha$, $\beta$, $N_a$ and $N_b$ to be constant over the whole recording and only expect $S$ and $C$ to change depending on the speech.

Based on the above model, estimates for $\alpha$, $\beta$, $N_a$, $N_b$ and the RMS levels of recorded blocks $C_a$ and $C_b$ we can solve for $S_a$ and $S_b$ which then allows us to derive dynamic noise levels that *include* crosstalk as follows:

```math
\begin{pmatrix}
T_a^2 \\ T_b^2
\end{pmatrix}
= \begin{bmatrix}
    0 & \alpha^2 \\
    \beta^2 & 0
\end{bmatrix}
\begin{pmatrix}
    S_a^2 \\ S_b^2
\end{pmatrix}
+ \begin{pmatrix}
    N_a^2 \\ N_b^2
\end{pmatrix}
```

These dynamic noise levels are then used as basis for speech activity thresholds for the noise gating effect. The following processing steps are applied to turn RMS levels into a smooth gain function with values between zero and one with which a recorded signal is going to be multiplied to mute and unmute sections:

- bidirectional hysteresis: When the ratio $C/T$ goes above a certain level, the gain is set to 1.0. When it falles below a certain level, the gain is set to 0.0. A gain of 1.0 is extended into both directions until the level falls below the lower limit.
- Small gaps (runs of zeros) in the gain function are closed to avoid muting sections below a certain duration
- The "gain envelope is widened" by adding fade-ins and fade-outs effectively limiting the rate of change
- The "gain envelope" is smoothed by a 3-tap lowpass filter followed by interpolation using a quadratic B-spline.
