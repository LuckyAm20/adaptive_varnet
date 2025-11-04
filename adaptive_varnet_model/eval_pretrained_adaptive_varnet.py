import json
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import pandas as pd
import lpips
import pydicom
from pydicom.dataset import FileDataset
import datetime
from pl_modules import AdaptiveVarNetModule, VarNetModule
from subsample import create_mask_for_mask_type

from fastmri import evaluate
from fastmri.data.mri_data import fetch_dir
from fastmri.data.transforms import MiniCoilTransform
from fastmri.pl_modules import FastMriDataModule


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def entropy(prob_mask: torch.Tensor):
    ent = -(prob_mask * prob_mask.log() + (1 - prob_mask) * (1 - prob_mask).log())
    ent[prob_mask == 0] = 0
    ent[prob_mask == 1] = 0
    return ent


def _to_lpips_input(img_np: np.ndarray) -> torch.Tensor:
    """
    img_np: (H,W) или (1,H,W). Приводим к (1,3,H,W) в диапазон [-1,1] для LPIPS.
    """
    x = img_np
    if x.ndim == 3:  # (1,H,W)
        x = x[0]
    m = float(np.max(x)) if np.max(x) > 0 else 1.0
    x = (x / m).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 2.0 - 1.0)[None, None, ...]  # (1,1,H,W)
    x = np.repeat(x, 3, axis=1)          # (1,3,H,W)
    return torch.from_numpy(x)


import mlflow
def _mlflow_start_if_enabled(args):
    if not getattr(args, "mlflow", False):
        return None
    try:
        tracking_uri = getattr(args, "mlflow_uri", None) or os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(getattr(args, "mlflow_experiment", "adaptive-varnet-eval"))
        run = mlflow.start_run(run_name=getattr(args, "mlflow_run_name", None))
        # базовые параметры запуска
        mlflow.log_params({
            "challenge": args.challenge,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_batches": args.num_batches,
            "compute_lpips": args.compute_lpips,
            "save_images": args.save_images,
            "save_images_dir": args.save_images_dir,
            "roi_csv": args.roi_csv or "",
            "mask_type": args.mask_type if hasattr(args, "mask_type") else "",
            "center_fractions": str(getattr(args, "center_fractions", "")),
            "accelerations": str(getattr(args, "accelerations", "")),
            "checkpoint": args.load_checkpoint,
        })
        return run
    except Exception as e:
        print(f"[MLflow] disabled in eval (reason: {e})")
        return None


def _mlflow_finish_and_log_artifacts(run, args, metrics_dict):
    if run is None:
        return
    try:
        # метрики
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float, np.floating)):
                mlflow.log_metric(k, float(v))

        if getattr(args, "mlflow_log_artifacts", True):
            if args.save_images and os.path.isdir(args.save_images_dir):
                mlflow.log_artifacts(args.save_images_dir, artifact_path="recon_vis")
            if args.roi_csv and os.path.exists(args.roi_csv):
                mlflow.log_artifact(args.roi_csv, artifact_path="inputs")
            if os.path.exists(args.load_checkpoint):
                mlflow.log_artifact(args.load_checkpoint, artifact_path="checkpoints")
    except Exception as e:
        print(f"[MLflow] artifact/metrics logging failed: {e}")
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass


def _crop_roi(img: np.ndarray, box):
    x, y, w, h = [int(v) for v in box]
    return img[..., y:y + h, x:x + w]


def _norm01(x: np.ndarray) -> np.ndarray:
    m = np.max(x) if np.max(x) > 0 else 1.0
    return np.clip(x / m, 0, 1)


def _save_pair(gt: np.ndarray, rec: np.ndarray, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _prep(x):
        # Убираем размерности 1 и гарантируем форму (H, W)
        x = np.squeeze(x)
        if x.ndim == 1:
            # если вдруг (H,) — разворачиваем в квадрат
            side = int(np.sqrt(x.shape[0]))
            x = x[:side * side].reshape(side, side)
        return _norm01(x)

    gt_img = _prep(gt)
    rec_img = _prep(rec)

    fig = plt.figure(figsize=(6, 3))
    ax1 = plt.subplot(1, 2, 1); ax1.imshow(gt_img, cmap="gray");  ax1.set_title("GT");    ax1.axis("off")
    ax2 = plt.subplot(1, 2, 2); ax2.imshow(rec_img, cmap="gray"); ax2.set_title("Recon"); ax2.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_dicom_pair(gt, rec, out_dir, base_name="recon"):
    import numpy as np
    import os
    import datetime
    import pydicom
    from pydicom.dataset import FileDataset

    os.makedirs(out_dir, exist_ok=True)

    def save_dicom(img_array, path):
        img = np.squeeze(img_array)
        if img.ndim != 2:
            raise ValueError(f"Unexpected image shape {img.shape}, expected 2D.")

        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_uint16 = np.uint16(img_norm * 65535)

        file_meta = pydicom.Dataset()
        ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = "MR"
        ds.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")
        ds.Rows, ds.Columns = img_uint16.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = img_uint16.tobytes()

        ds.save_as(path)

    gt_path = os.path.join(out_dir, f"{base_name}_gt.dcm")
    rec_path = os.path.join(out_dir, f"{base_name}_recon.dcm")

    # Сохраняем
    save_dicom(gt, gt_path)
    save_dicom(rec, rec_path)



def load_model(
    module_class: pl.LightningModule,
    fname: pathlib.Path,
):
    print(f"loading model from {fname}")
    checkpoint = torch.load(fname, map_location=torch.device("cpu"))

    # Initialise model with stored params
    module = module_class(**checkpoint["hyper_parameters"])

    # Load stored weights
    module.load_state_dict(checkpoint["state_dict"])

    return module


def cli_main(args):
    pl.seed_everything(0)
    run = _mlflow_start_if_enabled(args)
    # ------------
    # data
    # ------------
    mask = create_mask_for_mask_type(
        args.mask_type,
        args.center_fractions,
        args.accelerations,
        args.skip_low_freqs,
    )

    # 4-катушечная компрессия как в исходнике
    val_transform = MiniCoilTransform(
        mask_func=mask,
        num_compressed_coils=4,
        crop_size=args.crop_size,
    )

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------
    # model
    # ------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    non_adaptive = False
    try:
        print("Trying to load as AdaptiveVarNetModule...")
        model = load_model(AdaptiveVarNetModule, args.load_checkpoint)
        print("... Success!")
    except RuntimeError:
        print("Loading as AdaptiveVarNetModule failed, trying to load as VarNetModule...")
        model = load_model(VarNetModule, args.load_checkpoint)
        non_adaptive = True
        print("... Success!")

    model.to(device)

    data_loader = (
        data_module.val_dataloader()
        if args.data_mode == "val"
        else data_module.train_dataloader()
    )

    # ---- LPIPS ----
    lpips_net = None
    if args.compute_lpips:
        lpips_net = lpips.LPIPS(net=args.lpips_backbone)
        lpips_net.eval()

    # ---- ROI (для Box-SSIM) ----
    roi_map = {}
    if args.roi_csv:
        df = pd.read_csv(args.roi_csv)
        df["key"] = df["fname"].astype(str) + "|" + df["slice"].astype(str)
        roi_map = dict(zip(df["key"], zip(df["x"], df["y"], df["w"], df["h"])))

    # --------------------------------------------------------------------------------
    # Прогон датасета и накопление срезов по томам
    # --------------------------------------------------------------------------------
    vol_info = {}
    seen_slices = set()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if args.num_batches is not None and i == args.num_batches:
                break

            output, extra_outputs = model(
                batch.kspace.to(device),
                batch.masked_kspace.to(device),
                batch.mask.to(device),
            )

            prob_masks_list = extra_outputs["prob_masks"]
            if non_adaptive:
                # для обычного VarNet: одна маска
                assert len(prob_masks_list) == 1, "Expected one prob mask for non-adaptive VarNet"
                batch_prob_masks = prob_masks_list[0].squeeze().detach().cpu()
            else:
                # для AdaptiveVarNet: несколько масок
                print(
                    f"[Info] Detected {len(prob_masks_list)} policies in AdaptiveVarNet — averaging masks for visualization.")
                batch_prob_masks = torch.stack(prob_masks_list).mean(dim=0).squeeze().detach().cpu()
                if batch_prob_masks.ndim == 1:
                    batch_prob_masks = batch_prob_masks.unsqueeze(0)

            for bi, f in enumerate(batch.fname):
                if f not in vol_info:
                    vol_info[f] = []
                prob_mask = None if non_adaptive else batch_prob_masks[bi]
                slice_id = (f, batch.slice_num[bi])
                assert slice_id not in seen_slices
                seen_slices.add(slice_id)
                vol_info[f].append(
                    (
                        output[bi].cpu(),             # 0: recon (1,H,W)
                        batch.masked_kspace[bi].cpu(),# 1: masked kspace
                        batch.slice_num[bi],          # 2: slice index
                        batch.target[bi].cpu(),       # 3: target (1,H,W)
                        batch.max_value[bi],          # 4: max value
                        prob_mask,                    # 5: prob mask (128,) or None
                    )
                )

        # --------------------------------------------------------------------------------
        # Подсчёт метрик
        # --------------------------------------------------------------------------------
        all_prob_masks = []
        all_ssims, all_psnrs, all_nmses = [], [], []
        lpips_vals = []
        box_ssims = []

        # для сохранения картинок
        saved = 0
        save_dir = pathlib.Path(args.save_images_dir)

        for vol_name, vol_data in vol_info.items():
            # По-срезово для LPIPS/Box-SSIM/визуализаций
            for slice_data in vol_data:
                rec = slice_data[0].numpy()     # (1,H,W)
                gt  = slice_data[3].numpy()     # (1,H,W)
                sl  = int(slice_data[2])
                key = f"{pathlib.Path(vol_name).name}|{sl}"

                # классические метрики (по срезу)
                ssim = evaluate.ssim(gt[np.newaxis, ...], rec[np.newaxis, ...])
                psnr = evaluate.psnr(gt[np.newaxis, ...], rec[np.newaxis, ...])
                nmse = evaluate.nmse(gt[np.newaxis, ...], rec[np.newaxis, ...])
                all_ssims.append(ssim); all_psnrs.append(psnr); all_nmses.append(nmse)

                # LPIPS
                if lpips_net is not None:
                    r_t = _to_lpips_input(rec)
                    g_t = _to_lpips_input(gt)
                    with torch.no_grad():
                        v = lpips_net(g_t, r_t).item()
                    lpips_vals.append(v)

                # Box-SSIM (если задан ROI CSV)
                if key in roi_map:
                    box = roi_map[key]
                    rec_c = _crop_roi(rec, box)
                    gt_c  = _crop_roi(gt,  box)
                    box_ssims.append(evaluate.ssim(gt_c[np.newaxis, ...], rec_c[np.newaxis, ...]))

                if args.save_images and saved < args.max_save:
                    out_path = save_dir / f"{pathlib.Path(vol_name).stem}_slice{sl:03d}.png"
                    _save_pair(gt, rec, out_path)
                    saved += 1

                    if getattr(args, "save_dicom", False):
                        dicom_dir = save_dir / "dicom"
                        dicom_dir.mkdir(exist_ok=True, parents=True)
                        _save_dicom_pair(
                            gt,
                            rec,
                            dicom_dir,
                            base_name=f"{pathlib.Path(vol_name).stem}_slice{sl:03d}"
                        )

            # проб-маски (для отчёта энтропии/MI)
            if not non_adaptive:
                all_prob_masks.append(torch.stack([sd[-1] for sd in vol_data]))

        # агрегирование
        ssim_array = np.concatenate(np.array(all_ssims)[:, None], axis=0)
        psnr_array = np.concatenate(np.array(all_psnrs)[:, None], axis=0)
        nmse_array = np.concatenate(np.array(all_nmses)[:, None], axis=0)

        return_dict = {
            "ssim": ssim_array.mean().item(),
            "psnr": psnr_array.mean().item(),
            "nmse": nmse_array.mean().item(),
        }

        if lpips_vals:
            return_dict["lpips"] = float(np.mean(lpips_vals))
        if box_ssims:
            return_dict["box_ssim"] = float(np.mean(box_ssims))

        if all_prob_masks:
            prob_mask_tensor = torch.cat(all_prob_masks, dim=0).double()
            print(f"Computed {prob_mask_tensor.shape[0]} masks of size {prob_mask_tensor.shape[1]}")
            marg_prob = prob_mask_tensor.mean(dim=0, keepdim=True)
            marg_entropy = entropy(marg_prob).sum(dim=1)
            avg_cond_entropy = entropy(prob_mask_tensor).sum(dim=1).mean()
            mut_inf = marg_entropy - avg_cond_entropy
            return_dict.update(
                {
                    "cond_ent_ind": avg_cond_entropy.item(),
                    "marg_ent_ind": marg_entropy.item(),
                    "mi_ind": mut_inf.item(),
                }
            )

        try:
            with open(os.path.join(args.save_images_dir, "metrics.json"), "w") as f:
                json.dump(return_dict, f, indent=2)
        except Exception:
            pass

        print(return_dict)
        _mlflow_finish_and_log_artifacts(run, args, return_dict)


def build_args():
    parser = ArgumentParser()

    parser.add_argument("--load_checkpoint", type=pathlib.Path, help="Model checkpoint to load.")
    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float,
                        help=("Number of center lines to use in mask. 0.08 for acceleration 4, 0.04 for acceleration 8 models.",))
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int,
                        help=("Acceleration rates to use. For equispaced models set with center_fractions as in help.",))
    parser.add_argument("--crop_size", default=(128, 128), type=int, nargs="+", help="Crop size used by checkpoint.")
    parser.add_argument("--num_batches", default=None, type=int)
    parser.add_argument("--data_mode", default="val", type=str, choices=["train", "val"])
    parser.add_argument("--skip_low_freqs", default=True, type=str2bool,
                        help="Whether skip low-frequency lines when computing equispaced mask.")
    parser.add_argument("--vol_based", default=True, type=str2bool,
                        help="Whether to do volume-based evaluation (legacy flag; срезовые метрики всё равно считаются).")

    # расширенные метрики и визуализация
    parser.add_argument("--compute_lpips", default=True, type=str2bool, help="Подсчитывать LPIPS.")
    parser.add_argument("--lpips_backbone", default="alex", type=str, choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--roi_csv", default=None, type=str, help="CSV с ROI: columns=[fname,slice,x,y,w,h]")
    parser.add_argument("--save_images", default=True, type=str2bool, help="Сохранять пары GT/Recon.")
    parser.add_argument("--save_images_dir", default="./recon_vis", type=str)
    parser.add_argument("--max_save", default=12, type=int)
    parser.add_argument(
        "--save_dicom",
        action="store_true",
        help="If set, also save reconstructions and GT in DICOM format."
    )

    parser.add_argument("--mlflow", type=bool, default=True, help="Логировать метрики/артефакты в MLflow.")
    parser.add_argument("--mlflow_experiment", type=str, default="adaptive-varnet-eval", help="Эксперимент MLflow.")
    parser.add_argument("--mlflow_uri", type=str, default=None, help="override для MLFLOW_TRACKING_URI.")
    parser.add_argument("--mlflow_log_artifacts", type=bool, default=True,
                        help="Логировать артефакты (картинки, CSV, чекпоинт).")

    parser = AdaptiveVarNetModule.add_model_specific_args(parser)

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    data_path = fetch_dir("knee_path", path_config)

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        mask_type="adaptive_equispaced_fraction",
        challenge="multicoil",
        batch_size=64,
        test_path=None,
        num_workers=20,
    )

    args = parser.parse_args()
    assert (len(args.crop_size) == 2), f"Crop size must be of length 2, not {len(args.crop_size)}."
    return args


def run_cli():
    from dotenv import load_dotenv
    load_dotenv()
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()
