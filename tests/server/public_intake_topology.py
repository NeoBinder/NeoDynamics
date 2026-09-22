"""Synthetic metadata generation and independent complete topology read checks.

No simulation/DCD writer is imported. Only the generator child loads OpenMM;
its Context constructor is disabled before exporting the mass-only System.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

from neomd_server.contract import canonical_bytes, content_hash, validate

KINDS = ('atoms', 'residues', 'bonds', 'particles', 'topology_particle_indices')
DOCUMENT_BYTES = 1024**2
TOTAL_BYTES = 256 * 1024**2


class ArrayDigest:
    def __init__(self):
        self.digest = hashlib.sha256(b'[')
        self.count = 0

    def add(self, row):
        if self.count:
            self.digest.update(b',')
        self.digest.update(canonical_bytes(row))
        self.count += 1

    def finish(self):
        value = self.digest.copy()
        value.update(b']')
        return {'row_count': self.count, 'sha256': value.hexdigest()}


def source_evidence(directory):
    """Hash every original ordered row without reusing the page producer."""
    path = directory / 'manifest.json'
    if path.stat().st_size > TOTAL_BYTES:
        raise ValueError('source manifest exceeds fixture metadata budget')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    del raw
    collections = {}
    for kind in KINDS:
        digest = ArrayDigest()
        for row in manifest[kind]:
            digest.add(row)
        collections[kind] = digest.finish()
    value = {'manifest_sha256': sha, 'schema_version': manifest['schema_version'],
             'prepared_revision_id': manifest['prepared_revision_id'],
             'mapping_sha256': manifest['artifacts']['mapping.json']['sha256'],
             'particle_count': manifest['particle_count'],
             'topology_atom_count': manifest['topology_atom_count'], 'collections': collections}
    del manifest
    gc.collect()
    return value


def generate(directory, count, parent_pid):
    from neomd_server.processes import arm_parent_death

    arm_parent_death(parent_pid)
    import numpy as np
    import openmm
    from openmm import app, unit

    def forbidden(*_, **__):
        raise RuntimeError('metadata-only qualification forbids OpenMM Context')

    openmm.Context = forbidden
    from neomd.system import SystemBundle, export_system_bundle

    if not 1 <= count <= 100000:
        raise ValueError('synthetic metadata particle count outside approved bound')
    directory = Path(directory)
    directory.mkdir()
    topology, system = app.Topology(), openmm.System()
    chain = topology.addChain('Q')
    positions = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        positions[index] = (index % 100, (index // 100) % 100, index // 10000)
        topology.addAtom('C', app.element.carbon, topology.addResidue('SYN', chain, str(index + 1)))
        system.addParticle(12.0)
    positions *= .1
    source, xml = directory / 'source.pdbx', directory / 'system.xml'
    with source.open('w') as handle:
        app.PDBxFile.writeFile(topology, positions * unit.nanometer, handle)
    xml.write_text(openmm.XmlSerializer.serialize(system))
    exported = export_system_bundle(SystemBundle(topology_file=str(source), system_xml=str(xml)), directory / 'export')
    del topology, system, positions
    gc.collect()
    prepared = Path(exported.prepared_manifest).parent
    (directory / 'evidence.json').write_text(json.dumps(source_evidence(prepared), allow_nan=False))
    (directory / 'prepared.json').write_text(json.dumps({'relative_directory': str(prepared.relative_to(directory))}))


def strict_json(raw):
    if len(raw) > DOCUMENT_BYTES:
        raise ValueError('topology document exceeds byte budget')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate topology JSON field')
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    if canonical_bytes(value) != raw:
        raise ValueError('topology response bytes are not exact JCS')
    return value


class TopologyCheck:
    """Keep one page and compact digests; never assemble a second 100k-row DOM."""
    def __init__(self, raw, prepared, evidence):
        view = strict_json(raw)
        validate('TopologyView', view)
        if view['prepared'] != prepared or prepared['manifest_sha256'] != evidence['manifest_sha256']:
            raise ValueError('topology prepared binding differs from uploaded source')
        if any(prepared[key] != evidence[key] for key in ('prepared_revision_id', 'mapping_sha256')):
            raise ValueError('topology source revision/order differs')
        if (view['prepared_schema_version'] != evidence['schema_version']
                or any(view[key] != evidence[key] for key in ('particle_count', 'topology_atom_count'))
                or view['coordinate_unit'] != 'nm'):
            raise ValueError('topology source counts/version/unit differ')
        if content_hash({key: value for key, value in view.items() if key != 'view_sha256'}) != view['view_sha256']:
            raise ValueError('topology descriptor hash differs')
        self.view, self.evidence = view, evidence
        self.digests = {kind: ArrayDigest() for kind in KINDS}
        self.expected = []
        self.total_bytes = 0
        for collection, kind in zip(view['collections'], KINDS, strict=True):
            if collection['kind'] != kind or collection['row_count'] != evidence['collections'][kind]['row_count']:
                raise ValueError('topology collection inventory differs')
            first = 0
            for index, page in enumerate(collection['pages']):
                if page['page_index'] != index or page['first_row'] != first:
                    raise ValueError('topology page inventory has a gap/reorder')
                first += page['row_count']
                self.total_bytes += page['size_bytes']
                self.expected.append((kind, page))
            if first != collection['row_count']:
                raise ValueError('topology page inventory is incomplete')
        if len(self.expected) > 4096 or self.total_bytes > TOTAL_BYTES:
            raise ValueError('topology inventory exceeds aggregate budget')
        self.consumed = 0

    def page(self, raw):
        if self.consumed >= len(self.expected):
            raise ValueError('extra topology page')
        kind, spec = self.expected[self.consumed]
        if len(raw) != spec['size_bytes'] or hashlib.sha256(raw).hexdigest() != spec['content_sha256']:
            raise ValueError('topology page exact byte/hash mismatch')
        page = strict_json(raw)
        validate('TopologyPage', page)
        if (page['prepared'] != self.view['prepared'] or page['kind'] != kind
                or page['prepared_schema_version'] != self.view['prepared_schema_version']
                or any(page[key] != spec[key] for key in ('page_index', 'first_row'))
                or len(page['rows']) != spec['row_count']):
            raise ValueError('topology page identity differs')
        for row in page['rows']:
            self.digests[kind].add(row)
        self.consumed += 1

    def finish(self):
        if self.consumed != len(self.expected):
            raise ValueError('topology pages incomplete')
        actual = {kind: digest.finish() for kind, digest in self.digests.items()}
        if actual != self.evidence['collections']:
            raise ValueError('topology ordered full rows differ from source manifest')
        return {'view_sha256': self.view['view_sha256'], 'page_count': self.consumed,
                'page_bytes': self.total_bytes, 'collections': actual,
                'source_manifest_sha256': self.evidence['manifest_sha256'], 'all_source_rows_equal': True}


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        raise SystemExit('usage: public_intake_topology.py NEW_DIRECTORY PARTICLES PARENT_PID')
    generate(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
