"""
Unified Mechanistic Bioprocess Physics, Aquatic Chemistry, and Kinetics Core.

Single authoritative library imported by:
- code/models/hybrid_ode.py
- code/data/windowing.py
- code/environment/bioprocess_dynamics.py
- code/environment/pome_reactor_env.py
- DataSintesis/mock_data_generator.py

Literature References:
1. Benson, B. B., & Krause, D. (1984). Limnology and Oceanography, 29(3), 620-632.
   (Freshwater & saline dissolved oxygen solubility, virial & vapor pressure corrections)
2. Emerson, K., Russo, R. C., Lund, R. E., & Thurston, R. V. (1975).
   J. Fish. Res. Board Can., 32(12), 2379-2383. (NH3/NH4+ pKa(T) equilibrium)
3. Plummer, L. N., & Busenberg, E. (1982). Geochim. Cosmochim. Acta, 46(6), 1011-1040.
   (Carbonate thermodynamics pK1, pK2, pKH)
4. Bandura, A. V., & Lvov, S. N. (2006). J. Phys. Chem. Ref. Data, 35(1), 15-30.
   (Water ionization product pKw(T))
5. Davies, C. W. (1962) & Stumm, W., & Morgan, J. J. (1996). Aquatic Chemistry.
   (Davies activity corrections & Langelier ionic strength estimation)
6. Thimijan, R. W., & Heins, R. D. (1983). HortScience, 18(6), 818-822.
   (Photometric lux to PAR PPFD conversion for white/fluorescent-LED sources)
7. Rosso, L., Lobry, J. R., & Flandrois, J. P. (1993). J. Theor. Biol., 162(4), 447-463.
   (Cardinal Temperature Model with Inflection - CTMI)
8. Bernard, O., & Remond, B. (2012). Bioresour. Technol., 123, 520-527.
   (Microalgal light-temperature-self-shading Haldane kinetics)
9. Casagli, F., Zuccaro, G., Bernard, O., Steyer, J.-P., & Ficara, E. (2021).
   Water Research, 190, 116734. (ALBA microalgae-bacteria ecosystem model & pH solver)
10. Metcalf & Eddy / Tchobanoglous et al. (2014) & ASCE (2007).
    (Arrhenius aeration kLa temperature coefficient theta = 1.024)
11. Azov, Y., & Goldman, J. C. (1982). Appl. Environ. Microbiol., 43(4), 735-739.
    (Free-ammonia inhibition of microalgae)
12. Henze, M., Gujer, W., Mino, T., & van Loosdrecht, M. (2000). Activated Sludge Models.
    (Heterotrophic oxygen demand, biodegradable COD kinetics, endogenous respiration)
"""

from typing import Tuple, Union
import numpy as np
import torch

# ==============================================================================
# 1. Verified Physical & Photometric Constants
# ==============================================================================
LUX_TO_PPFD_LED: float = 0.0145  # umol photons m^-2 s^-1 per lux (Thimijan & Heins, 1983)
ARRHENIUS_KLA_THETA: float = 1.024  # Metcalf & Eddy / ASCE (2007)
P_ATM_STD_HPA: float = 1013.25
CO2_ATM_PARTIAL_ATM: float = 4.2e-4  # ~420 ppm atmospheric CO2
DIFFUSIVITY_RATIO_CO2_O2_SQRT: float = 0.90  # Danckwerts surface renewal sqrt(D_CO2/D_O2)

# Molar weights (g/mol = mg/mmol)
MW_C: float = 12.011
MW_CO2: float = 44.009
MW_N: float = 14.007
MW_O2: float = 31.998
MW_NAHCO3: float = 84.007

# Stoichiometric microalgal biomass carbon fixation (Chisti, 2007; Mirda et al., 2026)
# ~50% dry weight C => 1.83 g CO2 / g dry biomass fixed photoautotrophically
CO2_TO_BIOMASS_STOICH: float = 1.83

# Conservative organic fraction correction for POME TDS -> PSS salinity conversion
POME_INORGANIC_TDS_FRACTION: float = 0.65

ArrayLike = Union[float, int, np.ndarray]


# ==============================================================================
# 2. Light Conversion & Beer-Lambert Attenuation
# ==============================================================================
def lux_to_ppfd(lux: ArrayLike) -> np.ndarray:
    """Convert photobioreactor illuminance (lux) to PAR PPFD (umol m^-2 s^-1)."""
    return np.maximum(np.asarray(lux, dtype=float), 0.0) * LUX_TO_PPFD_LED


def lux_to_ppfd_torch(lux: torch.Tensor) -> torch.Tensor:
    """Differentiable lux -> PAR PPFD (umol m^-2 s^-1) conversion."""
    return lux.clamp(min=0.0) * LUX_TO_PPFD_LED


def beer_lambert_i_avg(
    irradiance_surface: ArrayLike,
    biomass_x: ArrayLike,
    k_par: float,
    depth_cm: float,
    k_bg: float = 0.0,
) -> np.ndarray:
    """
    Column-averaged irradiance I_avg across path depth under Beer-Lambert attenuation:
        tau = (k_par * ln(10) * X + k_bg) * depth_cm
        I_avg = I_0 * (1 - exp(-tau)) / tau
    """
    i0 = np.maximum(np.asarray(irradiance_surface, dtype=float), 0.0)
    x = np.maximum(np.asarray(biomass_x, dtype=float), 0.0)
    tau = np.maximum((k_par * np.log(10.0) * x + k_bg) * depth_cm, 1e-6)
    return i0 * (-np.expm1(-tau)) / tau


# ==============================================================================
# 3. Salinity, Ionic Strength, & Dissolved Oxygen Solubility (Benson & Krause 1984)
# ==============================================================================
def salinity_from_tds(
    tds_mg_l: ArrayLike,
    inorganic_fraction: float = POME_INORGANIC_TDS_FRACTION,
) -> np.ndarray:
    """
    Estimate practical salinity (PSU ~ g/kg) from TDS (mg/L) for mixed organic-inorganic
    effluents (UNESCO PSS-78 / APHA 2520B conductivity ratio).
    """
    tds = np.maximum(np.asarray(tds_mg_l, dtype=float), 0.0)
    return inorganic_fraction * tds / 1000.0


def salinity_from_tds_torch(
    tds_mg_l: torch.Tensor,
    inorganic_fraction: float = POME_INORGANIC_TDS_FRACTION,
) -> torch.Tensor:
    """Torch equivalent of salinity_from_tds."""
    return tds_mg_l.clamp(min=0.0) * (inorganic_fraction / 1000.0)


def ionic_strength_from_tds(
    tds_mg_l: ArrayLike,
    inorganic_fraction: float = POME_INORGANIC_TDS_FRACTION,
) -> np.ndarray:
    """
    Langelier (1936) ionic strength approximation I (mol/L):
        I = 2.5e-5 * TDS_inorganic (mg/L)
    """
    tds = np.maximum(np.asarray(tds_mg_l, dtype=float), 0.0)
    return 2.5e-5 * (inorganic_fraction * tds)


def _vapor_pressure_water_atm(t_k: Union[np.ndarray, torch.Tensor]):
    """Pure water saturation vapor pressure in atm (Benson & Krause 1984 / USGS 2011)."""
    if isinstance(t_k, torch.Tensor):
        return torch.exp(11.8571 - 3840.70 / t_k - 216961.0 / (t_k * t_k))
    return np.exp(11.8571 - 3840.70 / t_k - 216961.0 / (t_k * t_k))


def do_saturation_mgL(
    temp_c: ArrayLike,
    pressure_hpa: ArrayLike = P_ATM_STD_HPA,
    salinity_ppt: ArrayLike = 0.0,
) -> np.ndarray:
    """
    Dissolved oxygen equilibrium concentration (mg/L) per Benson & Krause (1984),
    Limnology and Oceanography 29(3): 620-632, with full atmospheric pressure and
    water-vapor non-ideality corrections (Check: 20 C, 0 PSU, 1013.25 hPa -> 9.092 mg/L).
    """
    t = np.asarray(temp_c, dtype=float)
    tk = t + 273.15
    p_atm = np.asarray(pressure_hpa, dtype=float) / P_ATM_STD_HPA
    sal = np.maximum(np.asarray(salinity_ppt, dtype=float), 0.0)

    ln_co = (
        -139.34411
        + 1.575701e5 / tk
        - 6.642308e7 / (tk**2)
        + 1.243800e10 / (tk**3)
        - 8.621949e11 / (tk**4)
    )
    sal_corr = sal * (0.017674 - 10.754 / tk + 2140.7 / (tk**2))
    c_zero = np.exp(ln_co - sal_corr)

    pwv = _vapor_pressure_water_atm(tk)
    theta = 0.000975 - 1.426e-5 * t + 6.436e-8 * (t**2)
    f_p = ((p_atm - pwv) * (1.0 - theta * p_atm)) / ((1.0 - pwv) * (1.0 - theta))
    return np.maximum(c_zero * np.maximum(f_p, 0.0), 0.0)


def do_saturation_torch(
    temp_c: torch.Tensor,
    pressure_hpa: torch.Tensor,
    salinity_ppt: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Benson & Krause (1984) DO saturation concentration (mg/L)."""
    tk = temp_c + 273.15
    p_atm = pressure_hpa / P_ATM_STD_HPA
    sal = salinity_ppt.clamp(min=0.0)

    ln_co = (
        -139.34411
        + 1.575701e5 / tk
        - 6.642308e7 / (tk * tk)
        + 1.243800e10 / (tk**3)
        - 8.621949e11 / (tk**4)
    )
    sal_corr = sal * (0.017674 - 10.754 / tk + 2140.7 / (tk * tk))
    c_zero = torch.exp(ln_co - sal_corr)

    pwv = _vapor_pressure_water_atm(tk)
    theta = 0.000975 - 1.426e-5 * temp_c + 6.436e-8 * (temp_c * temp_c)
    f_p = ((p_atm - pwv) * (1.0 - theta * p_atm)) / ((1.0 - pwv) * (1.0 - theta))
    return (c_zero * f_p.clamp(min=0.0)).clamp(min=0.0)


def kla_temp_factor(temp_c: ArrayLike) -> np.ndarray:
    """ASCE / Metcalf & Eddy Arrhenius gas transfer correction: 1.024^(T - 20)."""
    return ARRHENIUS_KLA_THETA ** (np.asarray(temp_c, dtype=float) - 20.0)


def kla_temp_factor_torch(temp_c: torch.Tensor) -> torch.Tensor:
    """Torch Arrhenius gas transfer correction: 1.024^(T - 20)."""
    return ARRHENIUS_KLA_THETA ** (temp_c - 20.0)


# ==============================================================================
# 4. Ammonia & Carbonate Speciation (Emerson 1975, Plummer & Busenberg 1982, Davies 1962)
# ==============================================================================
def davies_delta(temp_c: ArrayLike, ionic_strength: ArrayLike) -> np.ndarray:
    """
    Davies (1962) monovalent activity term Delta_I = -log10(gamma_1):
        Delta_I = A_DH(T) * (sqrt(I) / (1 + sqrt(I)) - 0.3 * I)
    """
    t = np.asarray(temp_c, dtype=float)
    ion = np.clip(np.asarray(ionic_strength, dtype=float), 0.0, 0.5)
    a_dh = 0.4918 + 6.614e-4 * t + 4.975e-6 * (t * t)
    sq = np.sqrt(ion)
    return a_dh * (sq / (1.0 + sq) - 0.3 * ion)


def free_nh3_fraction(
    temp_c: ArrayLike,
    ph: ArrayLike,
    ionic_strength: ArrayLike = 0.0,
) -> np.ndarray:
    """
    Emerson et al. (1975) unionized NH3 fraction of TAN, with optional Davies activity correction:
        pKa_0(T) = 0.09018 + 2729.92 / (T + 273.15)  [9.245 at 25 C]
        pKa' = pKa_0 + Delta_I  (mixed constant a_{H+}*[NH3]/[NH4+])
        phi = 1 / (1 + 10^(pKa' - pH))
    """
    tk = np.asarray(temp_c, dtype=float) + 273.15
    pka = 0.09018 + 2729.92 / tk + davies_delta(temp_c, ionic_strength)
    exponent = np.clip(pka - np.asarray(ph, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + 10.0**exponent)


def free_nh3_fraction_torch(
    temp_c: torch.Tensor,
    ph: torch.Tensor,
    ionic_strength: float = 0.0,
) -> torch.Tensor:
    """Differentiable Emerson et al. (1975) unionized NH3 fraction."""
    tk = temp_c + 273.15
    pka = 0.09018 + 2729.92 / tk
    if ionic_strength > 0.0:
        ion = min(max(float(ionic_strength), 0.0), 0.5)
        a_dh = 0.4918 + 6.614e-4 * temp_c + 4.975e-6 * (temp_c * temp_c)
        sq = np.sqrt(ion)
        pka = pka + a_dh * (sq / (1.0 + sq) - 0.3 * ion)
    exponent = (pka - ph).clamp(-30.0, 30.0)
    return 1.0 / (1.0 + torch.pow(10.0, exponent))


def carbonate_constants(
    temp_c: float,
    ionic_strength: float = 0.04,
) -> Tuple[float, float, float, float, float, float]:
    """
    Return apparent mixed equilibrium constants (referenced to pH = -log10(a_{H+}))
    and Davies Delta_I using Plummer & Busenberg (1982) + Bandura & Lvov (2006):
        (pK1_prime, pK2_prime, pKw_prime, pKH, pKa_nh_prime, delta_I)
    """
    tk = float(temp_c) + 273.15
    log_tk = np.log10(tk)

    # Plummer & Busenberg (1982) -log10(K) thermodynamic polynomials
    pk1_0 = -(-356.3094 - 0.06091964 * tk + 21834.37 / tk + 126.8339 * log_tk - 1684915.0 / (tk**2))
    pk2_0 = -(-107.8871 - 0.03252849 * tk + 5151.79 / tk + 38.92561 * log_tk - 563713.9 / (tk**2))
    pkh_0 = -(108.3865 + 0.01985076 * tk - 6919.53 / tk - 40.45154 * log_tk + 669365.0 / (tk**2))

    # Harned & Hamer (1933) / IUPAC standard water ionization pKw(T) for 0-60 C (~13.995 at 25 C)
    pkw_0 = 4470.99 / tk - 6.0875 + 0.017060 * tk
    pka_nh0 = 0.09018 + 2729.92 / tk

    di = float(davies_delta(temp_c, ionic_strength))
    # Mixed constants for species concentrations [X] with electrode activity a_H+:
    pk1_p = float(pk1_0 - di)
    pk2_p = float(pk2_0 - 3.0 * di)
    pkw_p = float(pkw_0 - di)
    pka_p = float(pka_nh0 + di)
    return pk1_p, pk2_p, pkw_p, float(pkh_0), pka_p, di


def carbonate_fractions(ph: float, pk1_p: float, pk2_p: float) -> Tuple[float, float, float]:
    """Return (alpha0_CO2, alpha1_HCO3, alpha2_CO3) at given pH and apparent pK1', pK2'."""
    x1 = 10.0 ** np.clip(ph - pk1_p, -20.0, 20.0)
    x2 = 10.0 ** np.clip(2.0 * ph - pk1_p - pk2_p, -25.0, 25.0)
    denom = 1.0 + x1 + x2
    return 1.0 / denom, x1 / denom, x2 / denom


def compute_conservative_charge_z(
    ph: float,
    dic_mol_l: float,
    tan_mol_l: float,
    temp_c: float,
    ionic_strength: float = 0.04,
) -> float:
    """
    Compute conservative strong-ion charge balance Z_net (eq/L) from observed state:
        Z_net = [HCO3-] + 2[CO3^2-] + [OH-] - [H+] - [NH4+]
    Inverts solve_ph exactly so t=0 simulated pH equals observed pH_0.
    """
    pk1_p, pk2_p, pkw_p, _, pka_p, di = carbonate_constants(temp_c, ionic_strength)
    _, a1, a2 = carbonate_fractions(ph, pk1_p, pk2_p)
    h_conc = 10.0 ** (-ph + di)
    oh_conc = 10.0 ** (ph - pkw_p)
    phi_nh3 = 1.0 / (1.0 + 10.0 ** np.clip(pka_p - ph, -30.0, 30.0))
    nh4_conc = max(tan_mol_l, 0.0) * (1.0 - phi_nh3)
    return float(dic_mol_l * (a1 + 2.0 * a2) + oh_conc - h_conc - nh4_conc)


def solve_ph(
    dic_mol_l: float,
    z_net_eq_l: float,
    tan_mol_l: float,
    temp_c: float,
    ionic_strength: float = 0.04,
    ph_bounds: Tuple[float, float] = (2.0, 12.5),
    tol: float = 1e-6,
) -> float:
    """
    Solve electroneutrality f(pH) = 0 for pH via strictly monotone safeguarded bisection
    (Casagli et al., 2021 ALBA / Stumm & Morgan, 1996):
        f(pH) = Z_net + [H+] + [NH4+] - [HCO3-] - 2[CO3^2-] - [OH-] = 0
    where df/dpH < 0 strictly across all pH.
    """
    dic = max(float(dic_mol_l), 1e-8)
    tan = max(float(tan_mol_l), 0.0)
    pk1_p, pk2_p, pkw_p, _, pka_p, di = carbonate_constants(temp_c, ionic_strength)

    def residual(ph_val: float) -> float:
        _, a1, a2 = carbonate_fractions(ph_val, pk1_p, pk2_p)
        h_conc = 10.0 ** (-ph_val + di)
        oh_conc = 10.0 ** (ph_val - pkw_p)
        nh4_conc = tan / (1.0 + 10.0 ** np.clip(ph_val - pka_p, -30.0, 30.0))
        return z_net_eq_l + h_conc + nh4_conc - dic * (a1 + 2.0 * a2) - oh_conc

    lo, hi = ph_bounds
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo <= 0.0:
        return lo
    if f_hi >= 0.0:
        return hi

    # ~30 bisection steps guarantee < 1e-8 pH residual cleanly and unconditionally
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if (hi - lo) < tol:
            return float(mid)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ==============================================================================
# 5. Biological Growth Inhibition & Kinetic Limitation Functions
# ==============================================================================
def f_light_haldane_bernard(
    i_avg_ppfd: ArrayLike,
    i_opt_ppfd: float,
    beta_i: float = 0.5,
) -> np.ndarray:
    """
    Bernard & Remond (2012) photoinhibition model:
        f_I(I) = (I / I_opt) / (I / I_opt + beta_i * (I / I_opt - 1)^2)
    Strictly equals 0.0 in complete darkness (I = 0) and peaks at 1.0 when I = I_opt.
    """
    r = np.maximum(np.asarray(i_avg_ppfd, dtype=float), 0.0) / max(float(i_opt_ppfd), 1e-6)
    denom = r + beta_i * ((r - 1.0) ** 2)
    return np.where(r <= 0.0, 0.0, r / np.maximum(denom, 1e-9))


def f_light_steele_torch(i_avg: torch.Tensor, i_opt: torch.Tensor) -> torch.Tensor:
    """Steele (1962) / Bernard photoinhibition curve with f_I(0) == 0."""
    r = i_avg.clamp(min=0.0) / i_opt.clamp(min=1e-6)
    return r * torch.exp(1.0 - r)


def ctmi_cardinal_np(
    val: ArrayLike,
    v_min: float,
    v_opt: float,
    v_max: float,
) -> np.ndarray:
    """
    Rosso et al. (1993) Cardinal Model with Inflection (CTMI) for temperature
    and pH CPM (Casagli et al., 2021). Returns 0.0 outside (v_min, v_max).
    """
    x = np.asarray(val, dtype=float)
    num = (x - v_max) * ((x - v_min) ** 2)
    den = (v_opt - v_min) * ((v_opt - v_min) * (x - v_opt) - (v_opt - v_max) * (v_opt + v_min - 2.0 * x))
    inside = (x > v_min) & (x < v_max)
    safe_den = np.where(np.abs(den) < 1e-9, 1.0, den)
    return np.where(inside, np.clip(num / safe_den, 0.0, 1.0), 0.0)


def ctmi_cardinal_torch(
    val: torch.Tensor,
    v_min: torch.Tensor,
    v_opt: torch.Tensor,
    v_max: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Rosso et al. (1993) CTMI cardinal curve."""
    num = (val - v_max) * ((val - v_min) ** 2)
    den = (v_opt - v_min) * ((v_opt - v_min) * (val - v_opt) - (v_opt - v_max) * (v_opt + v_min - 2.0 * val))
    inside = (val > v_min) & (val < v_max)
    safe_den = torch.where(den.abs() < 1e-9, torch.ones_like(den), den)
    return torch.where(inside, (num / safe_den).clamp(0.0, 1.0), torch.zeros_like(val))
