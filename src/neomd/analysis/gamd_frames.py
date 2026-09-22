"""OpenMM-free reading of the SDK's DCD saved-frame and particle-mass contract."""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from neomd.sinks import _AKMA_PS, DCD_HEADER_SIZE, dcd_frame_size, read_dcd_header

from .errors import AnalysisError


def read_gamd_frames(path, *, timestep_ps, masses=None, system_xml=None, bonds=None):
    """Read the SDK's DCD layout, including steps and triclinic boxes.

    Particle masses must be explicit or come from the prepared System XML;
    zero-mass virtual sites are retained. DCD carries float32 Angstrom positions
    and float32 AKMA timestep. The latter must agree with the provenance timestep
    to float32 precision; frame times then use that full-precision timestep.
    Periodic boxes use OpenMM's lower-triangular orientation convention.
    """
    try:
        path = Path(path)
        with path.open('rb') as stream:
            header = read_dcd_header(stream)
            if (not np.isfinite(timestep_ps) or timestep_ps <= 0
                    or not np.isclose(header.dt_akma * _AKMA_PS, timestep_ps,
                                      rtol=1e-7, atol=0)):
                raise AnalysisError('DCD timestep disagrees with GaMD provenance')
            expected = DCD_HEADER_SIZE + header.n_frames * dcd_frame_size(
                header.n_atoms, header.periodic)
            if path.stat().st_size != expected or header.n_frames == 0:
                raise AnalysisError('empty, torn or trailing-data DCD; finish/recover the run first')
            if masses is None:
                if system_xml is None:
                    raise AnalysisError('DCD analysis requires masses or prepared system_xml')
                root = ET.parse(system_xml).getroot()
                particles = root.find('Particles')
                if particles is None:
                    raise AnalysisError('prepared System XML has no Particles element')
                masses = [float(p.attrib['mass']) for p in particles.findall('Particle')]
            masses = np.asarray(masses, dtype=float)
            if (masses.shape != (header.n_atoms,) or not np.isfinite(masses).all()
                    or np.any(masses < 0)):
                raise AnalysisError('simulation particle masses do not match DCD atoms')
            positions = np.empty((header.n_frames, header.n_atoms, 3))
            boxes = np.empty((header.n_frames, 3, 3)) if header.periodic else None
            stream.seek(DCD_HEADER_SIZE)

            def record(length):
                if struct.unpack('<i', stream.read(4))[0] != length:
                    raise AnalysisError('invalid DCD record length')
                data = stream.read(length)
                if struct.unpack('<i', stream.read(4))[0] != length:
                    raise AnalysisError('invalid DCD closing record length')
                return data

            for frame in range(header.n_frames):
                if header.periodic:
                    a, cg, b, cb, ca, c = struct.unpack('<6d', record(48))
                    if not np.isfinite([a, b, c, ca, cb, cg]).all() or min(a, b, c) <= 0:
                        raise AnalysisError('invalid DCD box lengths')
                    sg2 = 1 - cg*cg
                    if sg2 <= 0:
                        raise AnalysisError('singular DCD box')
                    sg = np.sqrt(sg2)
                    cy = (ca - cb*cg)/sg
                    cz2 = 1 - cb*cb - cy*cy
                    if cz2 <= 0:
                        raise AnalysisError('singular DCD box')
                    boxes[frame] = np.asarray([[a, 0, 0], [b*cg, b*sg, 0],
                                              [c*cb, c*cy, c*np.sqrt(cz2)]]) * .1
                for axis in range(3):
                    positions[frame, :, axis] = np.frombuffer(
                        record(4*header.n_atoms), dtype='<f4').astype(float) * .1
        steps = header.first_step + np.arange(header.n_frames, dtype=np.int64)*header.interval_steps
        result = dict(positions_nm=positions, masses=masses, steps=steps,
                      time_ps=steps*timestep_ps)
        if boxes is not None:
            result['boxes_nm'] = boxes
        if bonds is not None:
            result['bonds'] = np.asarray(bonds)
        return result
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, struct.error) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError(f'cannot read GaMD trajectory: {error}', source=str(path)) from error
