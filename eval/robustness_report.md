# Output-parsing robustness

- Extraction on messy-but-recoverable outputs: **100%** (7 cases)
- Safe rejection of malformed outputs: **100%** (6 cases)

## Messy-but-recoverable (should extract the right prediction)

| perturbation | extracted |
|---|---|
| `clean` | yes |
| `prose_wrapped` | yes |
| `markdown_fence` | yes |
| `extra_keys` | yes |
| `whitespace_noise` | yes |
| `decoy_empty_object` | yes |
| `category_uppercase` | yes |

## Malformed (should return None, never guess)

| perturbation | safely rejected |
|---|---|
| `single_quotes` | yes |
| `truncated` | yes |
| `non_string_value` | yes |
| `missing_field` | yes |
| `empty` | yes |
| `prose_only` | yes |

Reproduce with `python eval/run_robustness.py`. The parser tolerates prose, markdown fences, extra keys and whitespace around a valid object, but treats single-quoted, truncated or wrong-typed output as invalid rather than guessing.
