from reader_utils import *
from skempi_lib import *


class SkempiReader(object):
    def __init__(self, producer, list_of_records, rotations, num_voxels=20):
        self.list_of_records = list_of_records
        self._p = np.random.permutation(len(list_of_records))
        self.nv = num_voxels
        self.rotations = rotations
        self.func = producer
        self.Q = deque()
        self.E = ThreadPoolExecutor(1)
        self.reset()

    def reset(self):
        self._pdb_ix = -1
        self.load_next_record()

    def load_next_record(self):
        self._pdb_ix += 1
        if self._pdb_ix == len(self._p):
            raise StopIteration

    @property
    def curr_record(self):
        return self.list_of_records[self._p[self._pdb_ix]]

    def read(self):
        self.load_next_record()
        self.E.submit(self.func, self.Q, self.curr_record, self.rotations, self.nv)


def non_blocking_producer_ddg_v1(queue, record, rotations, nv=20):
    onehot = get_counts(record)
    mut = record.mutations[0]
    res1 = record.struct[mut.chain_id][mut.i]
    res2 = record.mutant[mut.chain_id][mut.i]
    atoms1 = select_atoms_in_sphere(record.struct.atoms, res1.ca.coord, nv)
    atoms2 = select_atoms_in_sphere(record.mutant.atoms, res2.ca.coord, nv)
    for rot in rotations:
        voxels1 = get_4channel_voxels_around_res(atoms1, res1, rot, nv=nv)
        voxels2 = get_4channel_voxels_around_res(atoms2, res2, rot, nv=nv)
        queue.appendleft([voxels1, voxels2, onehot, record.ddg])


if __name__ == "__main__":
    records = load_skempi_v1()
    rotations = get_xyz_rotations(.25)
    reader = SkempiReader(non_blocking_producer_ddg_v1, records, rotations, num_voxels=20)
    loader = PdbLoader(reader, 50000, len(rotations))
    pbar = tqdm(range(len(loader)), desc="processing data...")
    for _, (v1, v2, oh, ddg) in enumerate(loader):
        pbar.update(1)
        msg = "qsize: %d" % (len(loader.reader.Q),)
        pbar.set_description(msg)
