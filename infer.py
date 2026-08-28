import click
import os
import yaml

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler


def sample_data(json_data):
    return (
        json_data["audio_duration"],
        json_data["prompt"],
        json_data["lyrics"],
        json_data["infer_step"],
        json_data["guidance_scale"],
        json_data["scheduler_type"],
        json_data["cfg_type"],
        json_data["omega_scale"],
        ", ".join(map(str, json_data["actual_seeds"])),
        json_data["guidance_interval"],
        json_data["guidance_interval_decay"],
        json_data["min_guidance_scale"],
        json_data["use_erg_tag"],
        json_data["use_erg_lyric"],
        json_data["use_erg_diffusion"],
        ", ".join(map(str, json_data["oss_steps"])),
        json_data.get("guidance_scale_text", 0.0),
        json_data.get("guidance_scale_lyric", 0.0),
    )


@click.command()
@click.option(
    "--checkpoint_path", type=str, default="", help="Path to the checkpoint directory"
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bfloat16")
@click.option(
    "--torch_compile", type=bool, default=False, help="Whether to use torch compile"
)
@click.option(
    "--cpu_offload",
    type=bool,
    default=False,
    help="Whether to use CPU offloading (only load current stage's model to GPU)",
)
@click.option(
    "--overlapped_decode",
    type=bool,
    default=False,
    help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)",
)
@click.option("--device_id", type=int, default=0, help="Device ID to use")
@click.option("--output_path", type=str, default=None, help="Path to save the output")
def main(
    checkpoint_path,
    bf16,
    torch_compile,
    cpu_offload,
    overlapped_decode,
    device_id,
    output_path,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # ----------------------------
    # 🧠 Try loading user YAML
    # ----------------------------
    yaml_path = os.getenv("ACE_CONFIG", "config/younging.yaml")

    cfg = None
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                print(f"✅ Loaded custom config: {yaml_path}")
        except Exception as e:
            print(f"⚠️  Failed to read YAML config: {e}")

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )
    print(model_demo)

    if cfg:
        # Use YAML settings
        prompt = cfg.get("genre", "") + ", " + cfg.get("mood", "")
        lyrics = cfg.get("text_input", "")
        infer_step = cfg.get("num_inference_steps", 60)
        guidance_scale = cfg.get("guidance_scale", 15)
        cfg_type = cfg.get("cfg_type", "apg")
        omega_scale = cfg.get("omega_scale", 10)
        manual_seeds = str(cfg.get("seed", 42))
        audio_duration = cfg.get("duration", 25.0)
        scheduler_type = "euler"

        model_demo(
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=0,
            use_erg_tag=True,
            use_erg_lyric=True,
            use_erg_diffusion=True,
            oss_steps="",
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            save_path=output_path,
        )
        print("🎵 Song generated from YAML config!")
        print(prompt)
        print(lyrics)
        return

    # ----------------------------
    # 🧩 Otherwise, fallback to random sample
    # ----------------------------
    print("⚠️  No YAML found, using DataSampler default...")
    data_sampler = DataSampler()
    json_data = data_sampler.sample()
    json_data = sample_data(json_data)
    print(json_data)

    (
        audio_duration,
        prompt,
        lyrics,
        infer_step,
        guidance_scale,
        scheduler_type,
        cfg_type,
        omega_scale,
        manual_seeds,
        guidance_interval,
        guidance_interval_decay,
        min_guidance_scale,
        use_erg_tag,
        use_erg_lyric,
        use_erg_diffusion,
        oss_steps,
        guidance_scale_text,
        guidance_scale_lyric,
    ) = json_data

    model_demo(
        audio_duration=audio_duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=omega_scale,
        manual_seeds=manual_seeds,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
        min_guidance_scale=min_guidance_scale,
        use_erg_tag=use_erg_tag,
        use_erg_lyric=use_erg_lyric,
        use_erg_diffusion=use_erg_diffusion,
        oss_steps=oss_steps,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        save_path=output_path,
    )


if __name__ == "__main__":
    main()
