import hashlib
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from petroscope.segmentation.utils.primary_augmentor import PrimaryAugmentor
from petroscope.segmentation.utils.data import avg_pool_2d, void_borders

from petroscope.segmentation.utils.vis import to_heat_map

from petroscope.segmentation.utils.base import UnitsFormatter


class DsCacher:

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.cache_dir.rmtree(ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, np.ndarray] | None:
        try:
            return np.load(self.cache_dir / key, allow_pickle=True)
        except Exception:
            return None

    def set(self, key: str, values: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.cache_dir / key, **values)

    @staticmethod
    def cache_key_mask(
        mask_path: Path,
        downscale: int,
        patch_s: int,
        extra="",
        hash_length: int = 8,
    ) -> str:
        hash = int(
            hashlib.sha256(
                str(mask_path.absolute()).encode("utf-8")
            ).hexdigest(),
            16,
        ) % (10**hash_length)
        key = f"{hash}_{patch_s}_{downscale}"
        if extra != "":
            key += "_" + extra

        return key + ".npz"


class DsItem:

    def __init__(
        self,
        img_path: Path,
        mask_path: Path,
        void_border_width: int,
    ) -> None:
        self.img_path = img_path
        self.mask_path = mask_path
        self.void_border_width = void_border_width
        self._load_image()

    def _load_image(self) -> None:
        self.image = np.array(Image.open(self.img_path), dtype=np.uint8)
        self.mask = np.array(Image.open(self.mask_path), dtype=np.uint8)
        if self.mask.ndim == 3:
            self.mask = self.mask[:, :, 0]

        if self.void_border_width > 0:
            void = void_borders(self.mask, border_width=self.void_border_width)
            self.mask_void = np.where(void == 0, 255, self.mask)
        else:
            self.mask_void = self.mask
        self.height, self.width = self.image.shape[:2]

        values, counts = np.unique(self.mask, return_counts=True)
        self.n_pixels = {v: c for v, c in zip(values, counts)}

    def load_prob_maps(
        self,
        patch_size: int,
        cls_indices: tuple[int],
        downscale: int | None,
        alpha: float,
        cacher: DsCacher | None,
    ) -> None:
        """Create prob maps for each class in the image (dataset item).

        Args:
            patch_size (int): patch size
            cls_indices (tuple[int]): indices of the classes to create maps
            downscale (int | None): downscale factor for the prob maps
            alpha (float): Power coeff for prob maps
            cacher (DsCacher): cacher for the prob maps
        """

        self.downscale = downscale if downscale is not None else 1
        self.alpha = alpha
        self.patch_size = patch_size
        self.p_maps = dict()
        cache_key = DsCacher.cache_key_mask(
            self.mask_path, self.downscale, patch_size
        )
        if cacher is not None:
            cached_p_maps = cacher.get(cache_key)
            if cached_p_maps is not None:
                self.p_maps = {int(k): m for k, m in cached_p_maps.items()}
                return
        self.p_maps = dict()
        for cls_idx in cls_indices:
            p_map = self._create_prob_map(
                self.mask, cls_idx, patch_size, self.downscale
            )
            if p_map is not None:
                self.p_maps[cls_idx] = p_map
        if cacher is not None:
            cacher.set(cache_key, {str(k): m for k, m in self.p_maps.items()})

        self._postprocess_prob_maps()

    def _postprocess_prob_maps(self) -> None:
        for k in self.p_maps:
            p_map = self.p_maps[k]
            p_map = p_map**self.alpha
            p_map /= np.sum(p_map)
            self.p_maps[k] = p_map

    def _create_prob_map(
        self,
        mask: np.ndarray,
        cls_idx: int,
        patch_size: int,
        downscale: int,
    ) -> np.ndarray | None:
        """calculates the prob map for the image

        Args:
            mask (np.ndarray): mask
            cls_idx (int): class index for which to calculate the prob map
            patch_size (int): patch size
            downscale (int): downscale factor (needed to decrease RAM usage)

        Returns:
            np.ndarray: prob map for the class
        """
        assert downscale >= 1
        s = patch_size // downscale
        self.patch_size = patch_size
        self.downscale = downscale
        self.patch_size_s = s
        self.height_s = self.height // downscale
        self.width_s = self.width // downscale

        mask_cls = np.where(mask == cls_idx, 1, 0).astype(np.float32)

        # if no pixels of this class in the image no need to build a map
        n_pixels = np.sum(mask_cls)
        if n_pixels == 0:
            return None

        # calc prob map for the defined patch size
        origin = (patch_size - 1) // 2
        p = uniform_filter(
            mask_cls, patch_size, mode="constant", cval=0.0, origin=-origin
        )

        # downscale the prob map
        if downscale > 1:
            p = avg_pool_2d(p, kernel_size=downscale)

        # normalize map
        max_p = np.max(p[:-s, :-s])
        min_p = np.min(p[:-s, :-s])
        p = (p - min_p) / (max_p - min_p)

        # fill right and bottom border with zeros
        p[-s:, :] = 0
        p[:, -s:] = 0

        # make it sum up to 1
        p = p / np.sum(p)

        # clean up
        del mask_cls

        return p

    def patch_random(
        self, trg_size: int = None
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        s = self.patch_size_s if trg_size is None else trg_size
        y = np.random.randint(low=0, high=self.height - s)
        x = np.random.randint(low=0, high=self.width - s)
        # extract image patch and mask patch
        patch_img = self.image[y : y + s, x : x + s, :]
        patch_mask = self.mask_void[y : y + s, x : x + s]
        return patch_img, patch_mask, (y, x)

    def patch_sampler(
        self, class_idx
    ) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:

        p = self.p_maps[class_idx]
        f = p.flatten()

        while True:
            pos = np.random.choice(p.size, p=f)
            y = pos // p.shape[1]
            x = pos % p.shape[1]

            # upscale coords
            if self.downscale > 1:
                x = x * self.downscale + np.random.randint(
                    low=0, high=self.downscale
                )
                y = y * self.downscale + np.random.randint(
                    low=0, high=self.downscale
                )

            # check if patch is out of bounds (it shouldn't happen)
            y = min(self.height - self.patch_size, y)
            x = min(self.width - self.patch_size, x)

            # extract image patch and mask patch
            patch_img = self.image[
                y : y + self.patch_size,
                x : x + self.patch_size,
                :,
            ]
            patch_mask = self.mask_void[
                y : y + self.patch_size,
                x : x + self.patch_size,
            ]
            yield patch_img, patch_mask, (y, x)

    def size_bytes(self):
        img_s = self.image.size * self.image.itemsize
        mask_s = self.mask.size * self.mask.itemsize
        p_maps_s = 0
        try:
            p_maps_s = sum(
                [p_map.size * p_map.itemsize for p_map in self.p_maps.values()]
            )
        except AttributeError:
            pass
        return img_s + mask_s + p_maps_s

    def size_patches_approx(self):
        return np.ceil(
            self.image.shape[0] * self.image.shape[1] / (self.patch_size**2)
        ).astype(int)

    @staticmethod
    def dict_key(p: Path) -> str:
        return str(p.absolute())


class DsAccumulator:
    def __init__(self, cls_indices: Iterable[int], store_history=True):
        self.cls_indices = list(cls_indices)
        self.reset()
        self.store_history = store_history

    def reset(self):
        self.accumulator = {i: 0 for i in self.cls_indices}
        self.cls_choice = {i: 0 for i in self.cls_indices}
        self.cls_choice[-1] = 0  # for random choice
        self._t_start = time.time()
        self.history = dict()

    def update(
        self,
        mask: np.ndarray,
        item_idx: int = None,
        pos: tuple[int, int] = None,
        is_random: bool = False,
    ):
        values, counts = np.unique(mask, return_counts=True)
        for v, c in zip(values, counts):
            if v not in self.accumulator:
                continue
            self.accumulator[v] += c
        if self.store_history and item_idx is not None and pos is not None:
            self.history[item_idx] = self.history.get(item_idx, []) + [pos]
        if is_random:
            self.cls_choice[-1] += 1

    def get_class_balanced(self, strict=False) -> int:
        if strict:
            cls_idx = min(self.accumulator, key=self.accumulator.get)
            self.cls_choice[cls_idx] += 1
            return cls_idx
        else:
            probs = np.array(
                [max(self.accumulator[i], 1) for i in self.cls_indices]
            )
            probs = (1 / probs) ** 2
            probs = probs / np.sum(probs)
            cls_idx = np.random.choice(np.array(self.cls_indices), p=probs)
            self.cls_choice[cls_idx] += 1
            return cls_idx

    def get_class_random(self) -> int:
        i = np.random.choice(len(self.cls_indices))
        idx = self.cls_indices[i]
        self.cls_choice[idx] += 1
        return idx

    def balancing_quality(self) -> float:
        v = self.accumulator.values()
        return max(v) / sum(v) - min(v) / sum(v)

    def __str__(
        self, labels: dict[int, str] = None, include_items=False
    ) -> str:
        pix_total = sum(self.accumulator.values())
        if pix_total == 0:
            return "Accumulator is empty"
        pix_total_s = UnitsFormatter.si(pix_total)
        items_total = sum(self.cls_choice.values())
        pixels_prc = [
            (i, self.accumulator[i] / pix_total * 100)
            for i in self.cls_indices
        ]
        pixels_prc_s = ", ".join([f"{i}: {prc:.1f}%" for i, prc in pixels_prc])
        classes_prc = [
            (i, self.cls_choice[i] / items_total * 100)
            for i in self.cls_indices
        ]
        classes_prc_s = ", ".join(
            [f"{i}: {prc:.1f}%" for i, prc in classes_prc]
        )

        items_prc = [
            len(self.history[item_idx]) / items_total * 100
            for item_idx in sorted(self.history.keys())
        ]
        items_prc_s = ", ".join(
            [f"{i}: {prc:.1f}%" for i, prc in enumerate(items_prc)]
        )

        performance = items_total / (time.time() - self._t_start)

        items_str = (
            f"\t items requested: {items_prc_s}\n" if include_items else ""
        )

        return (
            "Accumulator stats:\n"
            f"\t requests: {items_total}\n"
            f"\t pixels retrieved: {pix_total_s}\n"
            f"\t pixels per class: {pixels_prc_s}\n"
            f"\t classes requested: {classes_prc_s}\n"
            f"{items_str}"
            f"\t balancing quality: {self.balancing_quality():.2f}\n"
            f"\t performance: {performance:.2f} it/s"
        )


class SelfBalancingDataset:

    def __init__(
        self,
        img_mask_paths: Iterable[tuple[Path, Path]],
        cls_indices: tuple[int],
        patch_size: int,
        void_border_width: int = 0,
        balancing_strength: float = 0.8,
        class_area_consideration: float = 0.5,
        patch_positioning_accuracy: float = 0.5,
        acceleration: int | None = 8,
        augment_rotation: float | None = None,
        augment_scale: float | None = None,
        cache_dir: Path | None = None,
        print_class_distribution: bool = False,
    ) -> None:
        """
        The BalancedSegmDataset class is designed to extract patches from
        a collection of images and their corresponding masks in the task
        of segmentation. This class allows for flexible configuration,
        including balancing class distributions, considering the area of
        each class in the images, and augmenting patches with rotations
        and scaling. Additionally, it supports acceleration through
        downsampling of probability maps and caching to improve performance.
        It also supports a range of visualizations.

        Args:
            img_mask_paths (Iterable[tuple[Path, Path]]): An iterable
            containing pairs of image and mask paths.

            cls_indices (tuple[int]): A tuple of class indices used in the
            dataset. This can be a subset of all classes.

            void_border_width (int, optional): The width of border between
            classes which should not be considerated. This area in mask is
            filled with value 255. If set to 0, no border is added.

            patch_size (int): The desired size of each patch.

            balancing_strength (float, optional): A value between 0 and 1
            indicating the strength of balancing. If set to 1, patches are
            chosen with strong balancing (the minority class is chosen more
            frequently). If set to 0, class for each patch is selected
            randomly. Defaults to 0.8.

            class_area_consideration (float, optional): Defines how the area
            of each class in the image influences the selection of images.
            A value of 1 means the probability of choosing an image is directly
            proportional to the number of pixels of that class in the image.
            A value of -1 means the probability is inversely proportional.
            A value of 0 means the class area is not considered. Recommended
            range is [-1, 1]. Defaults to 0.5.

            patch_positioning_accuracy (float, optional): Controls the accuracy
            of patch positioning on the probability map. Higher values result
            in more accurate positioning. Range is [0, 1]. Defaults to 0.5.

            acceleration (int | None, optional): Sets the level of acceleration
            achieved by downsampling the probability maps. Higher values result
            in faster patch extraction and lower memory usage but decrease
            positioning accuracy. If no acceleration is needed, set to None.
            Possible values are [2, 4, 8, 16, 32]. Defaults to 8.

            augment_rotation (float | None, optional): Controls augmentation
            with random rotation. Larger rotation angles require extracting
            larger patches to ensure they can be cropped to the target size
            after rotation. Range is (0, 45]. If None, no rotation augmentation
            is performed. Defaults to 45.

            augment_scale (float | None, optional): Controls augmentation with
            random scale changes. The scale range is
            [1 / (1 + augment_scale), 1 + augment_scale]. Larger values require
            extracting larger patches. Range is (0, 0.5]. If None, no scale
            augmentation is performed. Defaults to 0.2.

            cache_dir (Path, optional): Path to the cache directory for storing
            probability maps and dataset cache. If None, no caching is used. It
            is highly recommended to set this to speed up patch extraction.
            Defaults to None.

            print_class_distribution (bool, optional): Whether to print
            the distribution of pixels per classes for this dataset.

        """

        # perform assertions
        assert patch_size > 0
        assert 0 <= balancing_strength <= 1
        # assert -1 <= class_area_consideration <= 1
        assert 0 <= patch_positioning_accuracy <= 1
        assert acceleration is None or acceleration in (2, 4, 8, 16, 32)
        assert augment_rotation is None or 0 < augment_rotation <= 45
        assert augment_scale is None or 0 < augment_scale <= 0.5

        # setup params
        self.cls_indices = cls_indices
        self.img_mask_paths = img_mask_paths
        self.void_border_width = void_border_width
        self.downscale_maps = acceleration
        self.balanced_strength = balancing_strength
        self.class_area_consideration = class_area_consideration
        self.patch_pos_acc = patch_positioning_accuracy
        self.print_class_distribution = print_class_distribution

        # setup supporting classes
        self.augmentor = PrimaryAugmentor(
            patch_size=patch_size,
            max_scale=augment_scale,
            max_rot_angle=augment_rotation,
        )
        self.visualizer = DsVisualizer(self)
        self.cacher = DsCacher(cache_dir)
        self.accum = None  # is set in _initialize

        # determine the patch source size depending on augmentation
        self.patch_size_src = self.augmentor.patch_size_src

        # perform initialization
        print("Initializing dataset...")
        t1 = time.time()
        self._initialize()
        t2 = time.time()
        print(f"initialization took {t2 - t1:.1f} seconds")
        print(
            f"Dataset size: {self.size()}, "
            f"approx len (num of patches): {len(self)}"
        )

    def _initialize(self) -> None:

        # create items
        self.items = [
            DsItem(img_p, mask_p, self.void_border_width)
            for img_p, mask_p in tqdm(self.img_mask_paths, "loading images")
        ]

        # pixels distribution is stored as nested dict:
        # cls_idx -> img_idx -> n_pixels
        self.ds_dstr = {cls_idx: dict() for cls_idx in self.cls_indices}

        # get prob maps (load or calculate)
        for i, item in enumerate(tqdm(self.items, "loading prob maps")):
            item.load_prob_maps(
                patch_size=self.patch_size_src,
                cls_indices=self.cls_indices,
                downscale=self.downscale_maps,
                alpha=self.patch_pos_acc,
                cacher=self.cacher,
            )
            # update pixel distribution
            for cls_idx, n in item.n_pixels.items():
                self.ds_dstr[cls_idx][i] = n

        # remove empty classes
        self.ds_dstr = {
            cls_idx: cls_dstr
            for cls_idx, cls_dstr in self.ds_dstr.items()
            if sum(cls_dstr.values()) > 0
        }

        # print disribution
        if self.print_class_distribution:
            for cls_idx, cls_dstr in self.ds_dstr.items():
                s = sum(cls_dstr.values())
                repr = [
                    f"{k}: {v * 100 / s:.1f}%" for k, v in cls_dstr.items()
                ]
                print(f"class {cls_idx}. Pixels: {repr}")

        self.accum = DsAccumulator(self.ds_dstr.keys())

    def _class_sampler(
        self, cls_idx: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:

        # get distribution for the class removing all empty items
        cls_dstr = {
            item_idx: n
            for item_idx, n in self.ds_dstr[cls_idx].items()
            if n > 0
        }
        items_indices = list(cls_dstr.keys())

        # Calculate weights according to pixels dstr and balancing coefficient:
        # 1. If coeff > 0 the more pixels of the class is in the image the more
        # likely it is to be chosen.
        # 2. If coeff = 0 then the images of the class are chosen with equal
        # probability.
        # 3. If coeff < 0 the less pixels of the class is in the image the more
        # likely it is to be chosen.
        weights = np.array(
            [
                pow(float(cls_dstr[i]), self.class_area_consideration)
                for i in items_indices
            ],
            dtype=np.float32,
        )
        weights /= np.sum(weights)

        # retrieve samplers for each dataset item
        samplers = {
            i: self.items[i].patch_sampler(cls_idx) for i in items_indices
        }

        while True:
            # choose item from the dataset
            item_idx = np.random.choice(items_indices, p=weights)
            img, mask, pos = next(samplers[item_idx])
            yield img, mask, item_idx, pos

    def random(self, update_accum=True) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a random patch and its corresponding mask from the dataset.
        """
        item_idx = np.random.choice(len(self.items))
        img, mask, pos = self.items[item_idx].patch_random(
            self.augmentor.patch_size_trg
        )
        if update_accum:
            self.accum.update(mask, item_idx, pos, is_random=True)
        return img, mask

    def sampler_random(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        while True:
            yield self.random(update_accum=False)

    def sampler_balanced(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        self.accum.reset()
        # prepare all class samplers
        samplers = {
            cls_idx: self._class_sampler(cls_idx)
            for cls_idx in self.ds_dstr.keys()
        }

        while True:
            # extract random patch with probability (1 - balancing_strength)
            if np.random.rand() > self.balanced_strength:
                yield self.random(update_accum=True)

            # extract balanced patch with probability balancing_strength
            cls_idx = self.accum.get_class_balanced()
            img, mask, item_idx, pos = next(samplers[cls_idx])
            img, mask, augm_code = self.augmentor.augment(img, mask)
            self.accum.update(mask, item_idx, pos)
            yield img, mask

    def __len__(self):
        return sum([item.size_patches_approx() for item in self.items])

    def __iter__(self):
        return self.sampler_balanced()

    def size(self) -> str:
        """
        Calculate the size of all items in the dataset in bytes and return it
        as a string formated with units (e.g. "100 KB").

        Returns:
            str: The size of all items in the dataset in bytes with units.
        """
        total_size = sum([dsc.size_bytes() for dsc in self.items])

        return UnitsFormatter.bytes(total_size)

    def visualize_accums(self, out_path: Path = Path("./out/accums/")):
        self.visualizer.visualize_accums(out_path)

    def visualize_probs(self, out_path, center_patch=True):
        self.visualizer.visualize_prob_maps(
            out_path, center_patch=center_patch
        )


class DsVisualizer:

    def __init__(self, ds: SelfBalancingDataset) -> None:
        self.ds = ds

    def visualize_prob_maps(
        self,
        out_path: Path,
        center_patch: bool,
        image_overlay: bool = True,
    ) -> None:
        out_path.mkdir(exist_ok=True, parents=True)
        for item in tqdm(self.ds.items, "visualizing probs"):
            for j, p_map in item.p_maps.items():
                if p_map is not None:
                    p1 = np.zeros_like(item.mask, dtype=np.float32)
                    p2 = (
                        p_map.repeat(self.ds.downscale_maps, axis=0).repeat(
                            self.ds.downscale_maps, axis=1
                        )
                        if self.ds.downscale_maps is not None
                        else p_map
                    )
                    p1[0 : p2.shape[0], 0 : p2.shape[1]] = p2
                    if center_patch:
                        p1 = np.roll(p1, self.ds.patch_size_src // 2, axis=0)
                        p1 = np.roll(p1, self.ds.patch_size_src // 2, axis=1)
                    heatmap = to_heat_map(p1 / np.max(p1))
                    Image.fromarray(heatmap).save(
                        out_path / f"{item.img_path.stem}_cl{j}.jpg"
                    )
                    if image_overlay:
                        overlay = (item.image * 0.5 + heatmap * 0.4).astype(
                            np.uint8
                        )
                        Image.fromarray(overlay).save(
                            out_path
                            / f"{item.img_path.stem}_cl{j}_overlay.jpg"
                        )

    def visualize_accums(self, out_path: Path) -> None:
        out_path.mkdir(exist_ok=True, parents=True)
        ps = self.ds.patch_size_src
        for i, item in enumerate(tqdm(self.ds.items, "visualizing accums")):
            m = np.zeros_like(item.mask, dtype=np.float32)
            if i not in self.ds.accum.history:
                heatmap = to_heat_map(m)
            else:
                for coord in self.ds.accum.history[i]:
                    m[coord[0] : coord[0] + ps, coord[1] : coord[1] + ps] += 1
                heatmap = to_heat_map(m / np.max(m))
            Image.fromarray(heatmap).save(
                out_path / f"{item.img_path.stem}.jpg"
            )
            overlay = (item.image * 0.5 + heatmap * 0.4).astype(np.uint8)
            Image.fromarray(overlay).save(
                out_path / f"{item.img_path.stem}_overlay.jpg"
            )
