from inverse_folding.utils.constants import ALL_CHOTHIA_KEYS


class ResidueNumberingEncoding:
    def __call__(
        self, resnum: int, insertion_code: str, chain_position: int, is_heavy_chain: bool
    ) -> int:
        raise NotImplementedError

    @property
    def embedding_size(self) -> int:
        raise NotImplementedError


class DummyEncoding(ResidueNumberingEncoding):
    @property
    def embedding_size(self) -> int:
        return 1

    def __call__(
        self, resnum: int, insertion_code: str, chain_position: int, is_heavy_chain: bool
    ) -> int:
        return 0


class ChainPositionEncoding(ResidueNumberingEncoding):
    def __init__(self, max_chain_length: int = 200) -> None:
        self.max_chain_length = max_chain_length

    @property
    def embedding_size(self) -> int:
        return self.max_chain_length

    def __call__(
        self, resnum: int, insertion_code: str, chain_position: int, is_heavy_chain: bool
    ) -> int:
        assert chain_position < self.max_chain_length, chain_position
        return chain_position


class ChothiaNumberingEncoding(ResidueNumberingEncoding):
    @property
    def embedding_size(self) -> int:
        return len(ALL_CHOTHIA_KEYS)

    def __call__(
        self, resnum: int, insertion_code: str, chain_position: int, is_heavy_chain: bool
    ) -> int:
        chain_prefix = "H" if is_heavy_chain else "L"
        key = chain_prefix + str(resnum) + insertion_code.strip()
        return ALL_CHOTHIA_KEYS[key]
