"""
Gymnasium Bioreactor Environment with Absorbing Violation Boundaries.
Simulates a 2.5 L POME photobioreactor cultivating Chlorella vulgaris.
Features:
- 5-dimensional continuous action space: aeration, sodium bicarbonate, microalgae inoculum, molasses, light
- Absorbing violation state semantics for safety (CMDP chance constraint formulation)
- Continuous state observation combining sensor dynamics and visual features.
(penjelasan.md §2 & §3.2, Chen & Luo, 2026).
"""

import os
import sys
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    ACTION_DIM,
    ACTION_COLUMNS,
    ACTION_BOUNDS,
    SENSOR_COLUMNS,
    RAW_POME_INIT,
    BIOLOGICAL_SAFETY_LIMITS as OPERATIONAL_CONSTRAINTS,
    REACTOR_VOLUME_LITERS,
    MAX_EPISODE_STEPS,
    COMPLIANCE_HOLD_STEPS,
    DECISION_HORIZON_HOURS,
)
from environment.bioprocess_dynamics import POMEBioprocessSimulator, POMEVisionGenerator
from data.windowing import compute_step_window_features
from environment.reward_rig import compute_rig_reward, compute_compliance_gap, compute_state_reward


class POMEReactorEnv(gymnasium.Env):
    """
    Continuous Bioreactor Environment for POME microalgae remediation.
    Implements standard Gymnasium step/reset interface with absorbing violation state.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        seed: int = 42,
        sensor_latent_dim: int = 64,
        visual_latent_dim: int = 32,
        max_steps: Optional[int] = None,
        decision_horizon_hours: float = DECISION_HORIZON_HOURS,
        vision_encoder: Optional[Any] = None,
    ):
        super().__init__()
        self.seed_val = seed
        self.decision_horizon_hours = float(decision_horizon_hours)
        if max_steps is None:
            self.max_steps = int(round(16.0 * 24.0 / self.decision_horizon_hours))
        else:
            self.max_steps = max_steps
        self.compliance_hold_steps = max(1, int(round(24.0 / self.decision_horizon_hours)))
        self.visual_latent_dim = visual_latent_dim

        # Action space: 5 continuous dosing levers
        # 1. aeration (0 - 6 h)
        # 2. sodium bicarbonate / nahco3 (0 - 5 g)
        # 3. microalgae inoculum / Chlorella vulgaris (0 - 100 mL)
        # 4. molasses carbon source (0 - 20 mL)
        # 5. light intensity (0 - 5000 lux)
        self.action_dim = ACTION_DIM
        low_acts = np.array([ACTION_BOUNDS[k][0] for k in ACTION_COLUMNS], dtype=np.float32)
        high_acts = np.array([ACTION_BOUNDS[k][1] for k in ACTION_COLUMNS], dtype=np.float32)
        self.action_space = spaces.Box(low=low_acts, high=high_acts, dtype=np.float32)

        # Observation space: 192 sensor statistical window features + visual latent vector (192 + 32 = 224)
        # Exactly matches offline corpus d_hist state dimensionality
        self.sensor_feature_dim = len(SENSOR_COLUMNS) * 12  # 192 descriptors
        self.obs_dim = self.sensor_feature_dim + visual_latent_dim  # 224
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        if vision_encoder is None:
            default_v_ckpt = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "checkpoints", "vision_encoder.pt")
            )
            if os.path.exists(default_v_ckpt):
                try:
                    from models.vision_encoder import VisionEncoder
                    v_ckpt = torch.load(default_v_ckpt, map_location="cpu", weights_only=False)
                    enc = VisionEncoder(latent_dim=int(v_ckpt.get("visual_dim", visual_latent_dim)))
                    enc.load_state_dict(v_ckpt["model_state_dict"])
                    enc.eval()
                    vision_encoder = enc
                except Exception:
                    vision_encoder = None

        self.sim = POMEBioprocessSimulator(seed=seed, decision_horizon_hours=self.decision_horizon_hours)
        self.vision_gen = POMEVisionGenerator(
            visual_dim=visual_latent_dim,
            seed=seed,
            vision_encoder=vision_encoder,
        )

        # Environment dynamics state variables
        self.current_state: Dict[str, float] = {}
        self.sensor_history: List[Dict[str, float]] = []
        self.biomass: float = 0.25
        self.alkalinity: float = 250.0
        self.step_count: int = 0
        self.compliant_consecutive_steps: int = 0
        self.is_terminal_violation: bool = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the bioreactor with fresh POME batch and initial microalgae culture."""
        super().reset(seed=seed)
        if seed is not None:
            self.sim = POMEBioprocessSimulator(seed=seed, decision_horizon_hours=self.decision_horizon_hours)

        self.current_state = dict(RAW_POME_INIT)
        for k in ["cod", "bod", "total_organic_carbon", "total_suspended_solids", "dissolved_ammonia"]:
            self.current_state[k] *= self.np_random.uniform(0.95, 1.05)

        init_biomass = 0.25
        if options and "initial_state" in options and isinstance(options["initial_state"], dict):
            ov = options["initial_state"]
            for k, v in ov.items():
                if k in self.current_state:
                    self.current_state[k] = float(v)
            # Keep stoichiometric consistency if caller only specified COD/TSS
            if "cod" in ov and "bod" not in ov:
                self.current_state["bod"] = 0.48 * self.current_state["cod"]
            if "cod" in ov and "total_organic_carbon" not in ov:
                self.current_state["total_organic_carbon"] = 0.316 * self.current_state["cod"]
            if "biomass" in ov:
                init_biomass = float(ov["biomass"])

        self.sensor_history = [dict(self.current_state)]
        self.biomass = init_biomass
        self.alkalinity = self.sim.reset_internal(self.current_state, self.biomass)
        self.step_count = 0
        self.compliant_consecutive_steps = 0
        self.is_terminal_violation = False

        obs = self._get_obs(is_violated=False)
        info = {
            "step": 0,
            "status": "initialized",
            "violation": False,
            "decision_horizon_hours": self.decision_horizon_hours,
            "raw_sensors": dict(self.current_state),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one control step (duration: decision_horizon_hours, default: 6 hours):
        Applies dosing action [aeration, sodium_bicarbonate, microalgae_inoculum, molasses, light_intensity].
        Terminates on absorbing violation or sustained compliance.
        """
        self.step_count += 1
        a_clipped = np.clip(action, self.action_space.low, self.action_space.high)
        action_dict = {col: float(a_clipped[i]) for i, col in enumerate(ACTION_COLUMNS)}

        # Absorbing state check: if previously collapsed, state remains in terminal violation
        if self.is_terminal_violation:
            obs = self._get_obs(is_violated=True)
            return obs, -10.0, True, False, {"violation": True, "reason": "absorbed"}

        # Store prior state for reward computation
        s_prev = dict(self.current_state)
        biomass_prev = float(self.biomass)

        # Step kinetics with dynamic decision horizon
        self.current_state, self.biomass, self.alkalinity = self.sim._step_kinetics(
            self.current_state,
            action_dict,
            self.biomass,
            self.alkalinity,
            collapsed=False,
            step=self.step_count,
            dt_hours=self.decision_horizon_hours,
        )
        self.sensor_history.append(dict(self.current_state))

        # Evaluate Tier-2 biological safety envelope (absorbing failure boundary c_t = 1)
        violation_reason = None
        if self.current_state["ph"] < OPERATIONAL_CONSTRAINTS["ph"]["min"]:
            violation_reason = "alkalinity_crash_ph_low"
        elif self.current_state["ph"] > OPERATIONAL_CONSTRAINTS["ph"]["max"]:
            violation_reason = "ph_too_high"
        elif self.current_state["water_temperature"] > OPERATIONAL_CONSTRAINTS["water_temperature"]["max"]:
            violation_reason = "thermal_limit_exceeded"
        elif self.current_state["water_temperature"] < OPERATIONAL_CONSTRAINTS["water_temperature"]["min"]:
            violation_reason = "thermal_limit_too_low"
        elif self.current_state["dissolved_oxygen"] < OPERATIONAL_CONSTRAINTS["dissolved_oxygen"]["min"]:
            violation_reason = "anoxic_collapse_do_low"

        c_t = 1.0 if violation_reason is not None else 0.0
        photo_gain_g = float(self.sim.last_photo_gain_gl * REACTOR_VOLUME_LITERS)

        if c_t == 1.0:
            # Transition directly into absorbing violation state
            self.is_terminal_violation = True
            obs = self._get_obs(is_violated=True)
            terminated = True
            truncated = False
            r_rig, info_rew = compute_rig_reward(
                s_prev,
                a_clipped,
                self.current_state,
                biomass_curr=biomass_prev,
                biomass_next=self.biomass,
                dt_hours=self.decision_horizon_hours,
                photo_biomass_gain_g=photo_gain_g,
            )
            info = {
                "violation": True,
                "reason": violation_reason,
                "absorbing_violation": True,
                "c_t": 1.0,
                "raw_sensors": dict(self.current_state),
                **info_rew,
            }
            return obs, -50.0 + r_rig, terminated, truncated, info

        # Check Tier-1 PerMenLH No. 5/2014 effluent discharge compliance
        _, is_ok = compute_compliance_gap(self.current_state)
        if is_ok:
            self.compliant_consecutive_steps += 1
        else:
            self.compliant_consecutive_steps = 0

        # Terminate if sustained compliance for 24h (dynamically scaled across horizons)
        is_compliant_terminal = bool(self.compliant_consecutive_steps >= self.compliance_hold_steps)
        is_max_steps = bool(self.step_count >= self.max_steps)

        terminated = is_compliant_terminal
        truncated = is_max_steps and not is_compliant_terminal

        # Compute explicit rig reward
        r_rig, info_rew = compute_rig_reward(
            s_prev,
            a_clipped,
            self.current_state,
            biomass_curr=biomass_prev,
            biomass_next=self.biomass,
            dt_hours=self.decision_horizon_hours,
            photo_biomass_gain_g=photo_gain_g,
        )

        obs = self._get_obs(is_violated=False)
        info = {
            "violation": False,
            "c_t": 0.0,
            "compliant": is_ok,
            "sustained_compliance": is_compliant_terminal,
            "biomass_gl": self.biomass,
            "step": self.step_count,
            "raw_sensors": dict(self.current_state),
            **info_rew,
        }

        return obs, r_rig, terminated, truncated, info

    def _get_obs(self, is_violated: bool = False) -> np.ndarray:
        """
        Construct multimodal observation s_t = [h_x, h_z] (224 dims):
        - h_x: 192 multi-resolution statistical descriptors from rolling sensor history (16 sensors * 12 stats)
        - h_z: 32 visual latent features from photography encoder
        Guarantees exact parity with offline training corpus d_hist.
        """
        h_x = compute_step_window_features(self.sensor_history, SENSOR_COLUMNS, window_size=4)

        # Generate corresponding visual latent
        _, h_z = self.vision_gen.generate_visual_state(
            cod=self.current_state.get("cod", 4000.0),
            ph=self.current_state.get("ph", 7.0),
            turbidity=self.current_state.get("turbidity", 500.0),
            step=self.step_count,
            is_violated=is_violated,
        )

        obs = np.concatenate([h_x, h_z], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
