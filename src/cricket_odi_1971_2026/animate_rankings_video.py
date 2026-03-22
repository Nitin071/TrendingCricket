"""
Animate Men's ODI batting top 15 as horizontal bars (length = ICC points).
Between months, each player's bar moves vertically (rank) and horizontally (points)
with smooth interpolation — bar chart race style.

Requires: pandas, matplotlib, Pillow (recommended). For MP4: ffmpeg on PATH.
  pip install pandas matplotlib pillow

Place West Indies crest at: assets/flags/west_indies.png (project root). Other flags are
cached from flagcdn into .cache/flags/ on first run.

Example:
  python animate_rankings_video.py --csv ../../odi_batting_rankings.csv -o odi_bat_top15.mp4
  python animate_rankings_video.py --start-year 2025 --end-year 2025 -o odi_bat_2025.gif

MP4 needs ffmpeg on PATH (macOS: brew install ffmpeg). If ffmpeg is missing, this script
falls back to GIF with the same basename.

zsh: do not paste comment lines that contain unquoted parentheses like (needs ffmpeg) —
zsh treats them as glob qualifiers. Use a plain comment or quote them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Writable config dir (helps in sandboxes / read-only home)
_mpl_cfg = Path(__file__).resolve().parents[2] / ".matplotlib"
_mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

# Rank 0 = #1 (top). Bars with y >= OFF_RANK slide off the chart (enter/exit).
OFF_RANK = 16.0
BAR_HEIGHT = 0.58

# ICC "Country" strings -> flagcdn.com ISO/subdivision codes (see https://flagcdn.com)
COUNTRY_TO_FLAG_CODE: dict[str, str] = {
    "Afghanistan": "af",
    "Australia": "au",
    "Bangladesh": "bd",
    "England": "gb-eng",
    "India": "in",
    "Ireland": "ie",
    "New Zealand": "nz",
    "Pakistan": "pk",
    "South Africa": "za",
    "Sri Lanka": "lk",
    "West Indies": "wi",
    "Zimbabwe": "zw",
}

FLAGCDN_PNG = "https://flagcdn.com/w80/{code}.png"
# West Indies: flagcdn has no WI; use bundled PNG under assets/flags/west_indies.png
WI_PLACEHOLDER = "wi"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def country_to_flag_code(name: str) -> str:
    n = (name or "").strip()
    if n in COUNTRY_TO_FLAG_CODE:
        return COUNTRY_TO_FLAG_CODE[n]
    # Loose match for minor spelling variants
    low = n.lower()
    for k, v in COUNTRY_TO_FLAG_CODE.items():
        if k.lower() == low:
            return v
    return "un"


def flag_cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / ".cache" / "flags"


def _png_header_ok(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def west_indies_bundled_png() -> Path | None:
    """Official WI crest shipped in the repo (user-provided PNG)."""
    for p in (
        project_root() / "assets" / "flags" / "west_indies.png",
        Path(__file__).resolve().parent / "assets" / "flags" / "west_indies.png",
    ):
        if p.is_file() and _png_header_ok(p):
            return p
    return None


def ensure_wi_placeholder(path: Path) -> None:
    """
    West Indies: flagcdn has no WI ISO code. Build a maroon badge with 'WI' text.
    (Wikimedia PNGs are often rate-limited; a local asset is reliable.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 200 and _png_header_ok(path):
        return

    wpx, hpx = 200, 150
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (wpx, hpx), (123, 0, 16))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 52
            )
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            except OSError:
                font = ImageFont.load_default()
        draw.text((wpx // 2, hpx // 2), "WI", fill=(255, 255, 255), font=font, anchor="mm")
        img.save(path, format="PNG")
        return
    except ImportError:
        pass

    # Fallback: uint8 RGB (matplotlib float RGBA imsave is easy to get wrong for imread)
    arr = np.zeros((hpx, wpx, 3), dtype=np.uint8)
    arr[:, :, 0] = 123
    arr[:, :, 1] = 0
    arr[:, :, 2] = 16
    mpimg.imsave(path, arr)


def _download_flag_code(cache: Path, code: str) -> Path | None:
    dest = cache / f"{code}.png"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = FLAGCDN_PNG.format(code=code)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TrendingCricket/1.0 (ranking animation)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 50:
            return None
        dest.write_bytes(data)
        return dest
    except (urllib.error.URLError, OSError):
        return None


def ensure_flag_png(country_name: str) -> Path | None:
    """
    Download or return cached PNG for country. Returns None if nothing could be stored.
    """
    code = country_to_flag_code(country_name)
    cache = flag_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{code}.png"

    if code == WI_PLACEHOLDER:
        bundled = west_indies_bundled_png()
        if bundled is not None:
            shutil.copy2(bundled, dest)
            return dest
        ensure_wi_placeholder(dest)
        return dest

    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    got = _download_flag_code(cache, code)
    if got is not None:
        return got

    if code != "un":
        return _download_flag_code(cache, "un")
    return None


def prefetch_flags(countries: list[str]) -> dict[str, Path | None]:
    """Resolve PNG path for each country name (uses cache)."""
    out: dict[str, Path | None] = {}
    seen: set[str] = set()
    for c in countries:
        if not c or c in seen:
            continue
        seen.add(c)
        out[c] = ensure_flag_png(c)
    return out


def preload_all_icc_flags() -> None:
    """Pre-download / copy every supported ICC country flag into .cache/flags."""
    cache = flag_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    for name in sorted(COUNTRY_TO_FLAG_CODE.keys()):
        ensure_flag_png(name)


def load_flag_image(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    try:
        return mpimg.imread(path)
    except OSError:
        return None


def ffmpeg_available() -> bool:
    """True if Matplotlib can use the ffmpeg writer (binary on PATH)."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        from matplotlib.animation import FFMpegWriter

        if hasattr(FFMpegWriter, "isAvailable"):
            return bool(FFMpegWriter.isAvailable())
    except Exception:
        pass
    return True


def load_and_filter(csv_path: Path, start_year: int | None, end_year: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
    df = df.dropna(subset=["Points", "Date"])
    if start_year is not None or end_year is not None:
        sy = start_year or 1971
        ey = end_year or 2099
        df = df[df["Date"].str.slice(0, 4).astype(int).between(sy, ey)]
    return df


def month_table(df: pd.DataFrame, date: str) -> pd.DataFrame:
    sub = df[df["Date"] == date].copy()
    sub = sub.sort_values("Points", ascending=False).head(15)
    return sub.reset_index(drop=True)


def _player_map(t: pd.DataFrame) -> dict[str, tuple[float, float, str]]:
    """player -> (rank_index 0..14, points, country)."""
    out: dict[str, tuple[float, float, str]] = {}
    for i in range(min(len(t), 15)):
        row = t.iloc[i]
        p = str(row["Player"])
        co = row["Country"] if "Country" in row.index else np.nan
        out[p] = (float(i), float(row["Points"]), "" if pd.isna(co) else str(co))
    return out


def snapshot_frame(t: pd.DataFrame) -> list[dict]:
    """One frame: horizontal bars at discrete ranks."""
    frame: list[dict] = []
    for i in range(min(len(t), 15)):
        row = t.iloc[i]
        frame.append({
            "y": float(i),
            "width": float(row["Points"]),
            "player": str(row["Player"]),
            "country": str(row.get("Country") or ""),
        })
    return frame


def ease_smoothstep(t: float) -> float:
    """Smooth 0..1 easing for more natural motion."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def interpolate_transition(
    t0: pd.DataFrame,
    t1: pd.DataFrame,
    t: float,
    *,
    use_easing: bool = True,
) -> list[dict]:
    """
    Interpolate between two monthly top-15 tables by player identity.
    y = vertical rank slot (0 = top); width = ICC points (bar length).
    """
    a = _player_map(t0)
    b = _player_map(t1)
    players = sorted(set(a.keys()) | set(b.keys()))
    te = ease_smoothstep(t) if use_easing else t

    frame: list[dict] = []
    for p in players:
        if p in a:
            r0, pts0, c0 = a[p]
        else:
            if p not in b:
                continue
            r1b, pts1b, c1b = b[p]
            r0, pts0, c0 = OFF_RANK, pts1b, c1b
        if p in b:
            r1, pts1, c1 = b[p]
        else:
            r1, pts1, c1 = OFF_RANK, pts0, c0

        if p not in a:
            pts0 = pts1
            c0 = c1
        if p not in b:
            pts1 = pts0
            c1 = c0

        y = (1.0 - te) * r0 + te * r1
        w = (1.0 - te) * pts0 + te * pts1
        country = c1 if te >= 0.5 else c0
        frame.append({
            "y": y,
            "width": max(0.0, w),
            "player": p,
            "country": country,
        })
    return frame


def make_frames(
    df: pd.DataFrame,
    dates: list[str],
    smooth_steps: int,
) -> tuple[list[list[dict]], float]:
    tables = [month_table(df, d) for d in dates]
    xmax = float(df["Points"].max()) * 1.08

    frames: list[list[dict]] = []

    if smooth_steps <= 1:
        for i in range(len(dates)):
            frames.append(snapshot_frame(tables[i]))
        return frames, xmax

    frames.append(snapshot_frame(tables[0]))
    for i in range(len(dates) - 1):
        for s in range(1, smooth_steps + 1):
            t = s / smooth_steps
            frames.append(
                interpolate_transition(tables[i], tables[i + 1], t, use_easing=True)
            )

    return frames, xmax


def frame_title(dates: list[str], frame_index: int, smooth_steps: int) -> str:
    if smooth_steps <= 1:
        return dates[min(frame_index, len(dates) - 1)]
    if frame_index == 0:
        return dates[0]
    trans = (frame_index - 1) // smooth_steps
    if trans >= len(dates) - 1:
        return dates[-1]
    step_in = (frame_index - 1) % smooth_steps
    t = (step_in + 1) / smooth_steps
    return dates[trans + 1] if t >= 0.5 else dates[trans]


def run(
    csv_path: Path,
    out_path: Path,
    fps: float,
    smooth_steps: int,
    start_year: int | None,
    end_year: int | None,
    dpi: int,
) -> None:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".mp4" and not ffmpeg_available():
        alt = out_path.with_suffix(".gif")
        print(
            "ffmpeg not found — cannot write MP4. Saving GIF instead:\n"
            f"  {alt}\n"
            "For MP4 on macOS:  brew install ffmpeg\n",
            file=sys.stderr,
        )
        out_path = alt

    df = load_and_filter(csv_path, start_year, end_year)
    if df.empty:
        raise SystemExit("No rows after filtering.")

    dates = sorted(df["Date"].unique())
    if len(dates) < 1:
        raise SystemExit("No dates in data.")

    frames, xmax = make_frames(df, dates, smooth_steps)

    print("Pre-downloading ICC flags (flagcdn + assets/flags/west_indies.png)…", file=sys.stderr)
    preload_all_icc_flags()
    countries_list = sorted(df["Country"].dropna().unique().tolist())
    flag_paths = prefetch_flags(countries_list)
    flag_rgba: dict[str, np.ndarray] = {}
    for c, pth in flag_paths.items():
        img = load_flag_image(pth)
        if img is not None:
            flag_rgba[c] = img
    print(f"Flags ready ({len(flag_rgba)} countries).", file=sys.stderr)

    BAR_FACE = "#333a4a"
    BAR_EDGE = "#484f58"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    title = ax.set_title("", fontsize=18, color="#e6edf3", pad=16)
    ax.set_xlabel("ICC rating points", fontsize=12, color="#8b949e")
    ax.set_ylabel("")
    ax.set_xlim(0, xmax)
    ax.set_ylim(OFF_RANK + 0.8, -0.6)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#8b949e")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", alpha=0.2, color="#30363d")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["left"].set_visible(False)

    def update(frame: int):
        ax.clear()
        ax.set_facecolor("#161b22")
        ax.set_xlim(0, xmax)
        ax.set_ylim(OFF_RANK + 0.8, -0.6)
        ax.tick_params(axis="x", colors="#8b949e")
        ax.tick_params(axis="y", colors="#8b949e")
        ax.grid(axis="x", alpha=0.2, color="#30363d")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.set_xlabel("ICC rating points", fontsize=12, color="#8b949e")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)

        data = frames[frame]
        for b in data:
            w = b["width"]
            if w < 0.01:
                continue
            y = b["y"]
            y0 = y - BAR_HEIGHT / 2
            y1 = y + BAR_HEIGHT / 2

            ax.barh(
                y,
                w,
                height=BAR_HEIGHT,
                left=0,
                color=BAR_FACE,
                edgecolor=BAR_EDGE,
                linewidth=0.6,
                alpha=0.98,
                zorder=1,
            )

            country = b.get("country") or ""
            img = flag_rgba.get(country)
            flag_w = 0.0
            if img is not None:
                flag_w = min(w * 0.26, xmax * 0.072, max(w * 0.92, 0.0))
                flag_w = max(flag_w, 0.0)
                if flag_w > xmax * 0.003:
                    ax.imshow(
                        img,
                        aspect="auto",
                        extent=[0, flag_w, y0, y1],
                        origin="upper",
                        zorder=5,
                        clip_on=True,
                    )

            pts_txt = f"{w:.0f}"
            pad = xmax * 0.008
            min_room = flag_w + pad * 2
            pts_inside = w > min_room + xmax * 0.02
            if pts_inside:
                ax.text(
                    w - pad,
                    y,
                    pts_txt,
                    va="center",
                    ha="right",
                    fontsize=10,
                    color="#f0f6fc",
                    fontweight="semibold",
                    zorder=7,
                )
            else:
                ax.text(
                    w + xmax * 0.01,
                    y,
                    pts_txt,
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#8b949e",
                    zorder=7,
                )

            label = b["player"]
            if len(label) > 22:
                label = label[:21] + "…"
            name_x = w + (xmax * 0.018 if pts_inside else xmax * 0.095)
            ax.text(
                name_x,
                y,
                label,
                va="center",
                ha="left",
                fontsize=9,
                color="#c9d1d9",
                zorder=6,
            )

        label_date = frame_title(dates, frame, smooth_steps)
        title_obj = ax.set_title(
            f"Men's ODI batting — top 15 — {label_date}",
            fontsize=18,
            color="#e6edf3",
            pad=16,
        )
        return [title_obj]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )

    if out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    else:
        try:
            anim.save(
                out_path,
                writer="ffmpeg",
                fps=fps,
                dpi=dpi,
                codec="libx264",
                bitrate=8000,
            )
        except Exception as e:
            alt = out_path.with_suffix(".gif")
            print(
                f"MP4 save failed ({e}). Retrying as GIF: {alt}",
                file=sys.stderr,
            )
            anim.save(alt, writer="pillow", fps=fps, dpi=dpi)
            out_path = alt

    plt.close(fig)
    print(f"Saved: {out_path} ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Horizontal bar animation of ODI batting top 15 (rank moves vertically)."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "odi_batting_rankings.csv",
        help="Path to odi_batting_rankings.csv",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("odi_bat_top15.mp4"))
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument(
        "--smooth",
        type=int,
        default=12,
        metavar="N",
        help="Frames per month transition (vertical/horizontal motion). 1 = one frame per month, no tween.",
    )
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    run(
        csv_path=args.csv,
        out_path=args.output,
        fps=args.fps,
        smooth_steps=max(1, args.smooth),
        start_year=args.start_year,
        end_year=args.end_year,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
