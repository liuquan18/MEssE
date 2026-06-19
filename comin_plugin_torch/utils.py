import datetime
import torch


def parse_icon_datetime(iso_str: str) -> datetime.datetime:
    """Parse the ISO 8601 string returned by comin.current_get_datetime()."""
    clean = str(iso_str).split(".")[0].rstrip("Z")
    dt = datetime.datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=datetime.timezone.utc)



def to_hpx_faces(owned_vals: torch.Tensor, hpx_level) -> torch.Tensor:
    """Reshape (n_owned_pixels, nlev) to (faces_per_rank, nlev, nside, nside)."""
    nside = 2**hpx_level
    n_owned, nlev = owned_vals.shape
    faces = n_owned // (nside * nside)

    return (
        owned_vals.reshape(faces, nside, nside, nlev).permute(0, 3, 1, 2).contiguous()
    )


def from_hpx_faces(pred_faces: torch.Tensor) -> torch.Tensor:
    """Reshape (faces_per_rank, nlev, nside, nside) back to (n_owned_pixels, nlev).

    Inverse of to_hpx_faces.
    """
    return pred_faces.permute(0, 2, 3, 1).contiguous().reshape(-1, pred_faces.shape[1])

