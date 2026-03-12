from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

Matrix = dict[str, dict[int, float]]


def score_to_float(value) -> float | None:
    if value == "C":
        return 1.0
    if value == "I":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_latest_logs(
    log_dir: str, benchmark: str, expected_samples: int = 100
) -> list[EvalLog]:
    all_infos = list_eval_logs(log_dir)
    infos = [i for i in all_infos if i.task == benchmark]

    latest: dict[tuple[str, str], tuple[str, str]] = {}
    for info in infos:
        log = read_eval_log(info.name, header_only=True)
        if log.status != "success" or not log.results:
            continue
        if log.results.completed_samples != expected_samples:
            continue
        config_json = log.eval.model_generate_config.model_dump_json(exclude_none=True)
        key = (log.eval.model, config_json)
        created = log.eval.created
        if key not in latest or created > latest[key][0]:
            latest[key] = (created, info.name)

    return [read_eval_log(name) for _, name in latest.values()]


def build_matrix(
    logs: list[EvalLog], name_lookup: dict
) -> tuple[Matrix, dict[int, dict]]:
    matrix: Matrix = {}
    question_meta: dict[int, dict] = {}

    for log in logs:
        config_json = log.eval.model_generate_config.model_dump_json(exclude_none=True)
        key = (log.eval.model, config_json)
        model_name = name_lookup.get(key, log.eval.model)

        scores: dict[int, float] = {}
        for sample in log.samples or []:
            qid = sample.id
            score_obj = next(iter(sample.scores.values())) if sample.scores else None
            if score_obj is None:
                continue
            val = score_to_float(score_obj.value)
            if val is not None:
                scores[qid] = val
            if qid not in question_meta and sample.metadata:
                question_meta[qid] = sample.metadata

        matrix[model_name] = scores

    return matrix, question_meta
