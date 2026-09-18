"""Pre-deserialization allowlist for remotely supplied core OpenMM Systems."""

from pathlib import Path
from xml.etree import ElementTree

from .contract import ContractError

CORE_XML_TYPES = frozenset(
    {
        "System",
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "RBTorsionForce",
        "NonbondedForce",
        "CustomBondForce",
        "CustomAngleForce",
        "CustomTorsionForce",
        "CustomExternalForce",
        "CustomNonbondedForce",
        "CustomCompoundBondForce",
        "CustomCentroidBondForce",
        "CustomCVForce",
        "CustomGBForce",
        "CustomHbondForce",
        "CustomManyParticleForce",
        "GBSAOBCForce",
        "CMMotionRemover",
        "MonteCarloBarostat",
        "MonteCarloAnisotropicBarostat",
        "MonteCarloMembraneBarostat",
        "RMSDForce",
        "CMAPTorsionForce",
        "GayBerneForce",
        "Continuous1DFunction",
        "Continuous2DFunction",
        "Continuous3DFunction",
        "Discrete1DFunction",
        "Discrete2DFunction",
        "Discrete3DFunction",
    }
)


def validate_system_xml(path, *, max_bytes, max_forces):
    """Reject plugins/loaders/entities before any OpenMM deserialization occurs."""
    path = Path(path)
    problems = []
    if path.stat().st_size > max_bytes:
        raise ContractError([("/system", "serialized System exceeds the configured byte limit")])
    data = path.read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ContractError([("/system", "XML document declarations/entities are forbidden")])
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise ContractError([("/system", "invalid serialized System XML")]) from error
    if root.tag != "System" or root.attrib.get("type") != "System":
        problems.append(("/system", "the root must be a core OpenMM System"))
    forces = 0
    for node in root.iter():
        if node.tag == "Force":
            forces += 1
        declared = node.attrib.get("type")
        if declared is not None and declared not in CORE_XML_TYPES:
            problems.append(("/system", "serialized object type is outside the core allowlist"))
        if any(
            key.lower() in {"file", "filename", "path", "url", "module", "library", "model_file"}
            for key in node.attrib
        ):
            problems.append(("/system", "serialized file/plugin loaders are forbidden"))
    if forces > max_forces:
        problems.append(("/system", "force inventory exceeds the configured limit"))
    if problems:
        raise ContractError(problems)
