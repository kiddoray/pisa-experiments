import sys; sys.path = ["kubric"] + sys.path
import pathlib
from kubric.assets.asset_preprocessing import convert_objs_to_kubric


if __name__ == "__main__":
    list_dir = '/mnt/solo/kiddoray/omniobject3d/omniobject3d___OmniObject3D-New/raw/raw_scans'
    obj_paths = [str(p) for p in pathlib.Path(list_dir).rglob('*.obj')]
    convert_objs_to_kubric(
        obj_paths=obj_paths,
        output_folder="/mnt/solo/kiddoray/kubric_assets/omniobject3d"
    )