from __future__ import annotations

import pandas as pd


def default_utility_table(
    subset_docnos,
    profiles,
    full_profile,
    *,
    default_utility_profile_name: str | None,
    tail_lookup=None,
) -> pd.DataFrame:
    aggregated_rows = []
    for docno in subset_docnos:
        for profile in profiles:
            default_utility = fallback_utility_for_profile(
                str(docno),
                profile.name,
                full_profile.name,
                default_utility_profile_name=default_utility_profile_name,
                tail_lookup=tail_lookup,
            )
            aggregated_rows.append({"docno": str(docno), "profile": profile.name, "utility": default_utility})
    return pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"])


def aggregate_utility_table(
    subset_docnos,
    profiles,
    full_profile,
    per_doc_profile_utilities_sum,
    per_doc_profile_utilities_count,
    *,
    default_utility_profile_name: str | None,
    tail_lookup=None,
) -> tuple[pd.DataFrame, int]:
    aggregated_rows = []
    default_count = 0
    for docno in [str(d) for d in subset_docnos]:
        has_any = False
        for profile in profiles:

            agg_value = per_doc_profile_utilities_sum.get((docno, profile.name))
            if agg_value is not None:
                count = per_doc_profile_utilities_count.get((docno, profile.name), 1)
                agg = float(agg_value) / count
                has_any = True
            else:
                agg = fallback_utility_for_profile(
                    docno,
                    profile.name,
                    full_profile.name,
                    default_utility_profile_name=default_utility_profile_name,
                    tail_lookup=tail_lookup,
                )
            aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": agg})
        if not has_any:
            default_count += 1
    return pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"]), default_count


def fallback_utility_for_profile(
    docno: str,
    profile_name: str,
    full_profile_name: str,
    *,
    default_utility_profile_name: str | None,
    tail_lookup,
) -> float:
    if tail_lookup:
        value = tail_lookup.get((str(docno), profile_name))
        if value is not None:
            return float(value)
    return default_utility_for_profile(
        profile_name,
        full_profile_name,
        default_utility_profile_name=default_utility_profile_name,
    )


def default_utility_for_profile(
    profile_name: str,
    full_profile_name: str,
    *,
    default_utility_profile_name: str | None,
) -> float:
    preferred_profile = default_utility_profile_name or full_profile_name
    return 1.0 if profile_name == preferred_profile else 0.0


def tail_lookup_from_table(tail_table_df: pd.DataFrame) -> dict:
    if tail_table_df.empty:
        return {}
    return {
        (str(row["docno"]), str(row["profile"])): float(row["utility"])
        for row in tail_table_df.loc[:, ["docno", "profile", "utility"]].to_dict(orient="records")
    }
