# Local API

The API is single-user and localhost-only. Errors use
`{"error": "stable_code", "detail": "actionable message"}`; asynchronous jobs
also expose `error_code`.

| Endpoint | Purpose |
|---|---|
| `GET /api/system` | Runtime device, memory, and versions |
| `GET /api/algorithms` | Algorithm parameter schema |
| `POST /api/estimate` | Approximate memory/time planning estimate |
| `GET /api/fs/list` | Server-side folder listing |
| `POST /api/fs/pick` | Native file/folder picker |
| `GET /api/projects` | List registered projects |
| `POST /api/projects` | Create a project or scratch project |
| `GET /api/projects/{pid}` | Read a project |
| `PATCH /api/projects/{pid}` | Rename/save a project |
| `DELETE /api/projects/{pid}` | Unregister without deleting user files |
| `PATCH /api/projects/{pid}/ui-state` | Persist opaque workspace state |
| `POST /api/projects/{pid}/frames/scan` | Replace inputs from a folder and validate |
| `POST /api/projects/{pid}/frames/add` | Add folders/files and validate |
| `POST /api/projects/{pid}/frames/remove` | Remove input references |
| `POST /api/projects/{pid}/frames/auto-group` | Propose batch groups |
| `GET /api/projects/{pid}/frame-thumb` | Display-only source thumbnail |
| `POST /api/projects/{pid}/jobs` | Enqueue `select`, `stack`, or `export` |
| `POST /api/projects/{pid}/export` | Enqueue a domain-aware export |
| `GET /api/jobs` | Queue and history |
| `GET /api/jobs/{jid}` | Job state/result/error |
| `DELETE /api/jobs/{jid}` | Cancel pending/running job |
| `POST /api/jobs/reorder` | Reorder pending job ids |
| `GET /api/presets` | List global presets |
| `POST /api/presets` | Save a preset |
| `DELETE /api/presets/{name}` | Delete a preset |
| `POST /api/projects/{pid}/viewer/register` | Register/build tiles for an image |
| `GET /api/viewer/{image_id}/tile/{z}/{x}/{y}` | Read a 256-pixel viewer tile |
| `GET /api/viewer/{image_id}/analysis` | RGB/luminance histogram and clipping counts |
| `GET /api/projects/{project_id}/retouch` | List retouch sessions |
| `POST /api/projects/{project_id}/retouch` | Create/reopen retouch session |
| `DELETE /api/retouch/{session_id}` | Delete retouch session |
| `POST /api/retouch/{session_id}/stroke` | Apply a stroke |
| `POST /api/retouch/{session_id}/undo` | Undo stroke |
| `POST /api/retouch/{session_id}/redo` | Redo stroke |
| `POST /api/retouch/{session_id}/flatten` | Publish a named flattened result |

## Job parameters

Stack jobs accept `method`, `device`, `algo_params`, optional `frames`, `align`, and
`select`. Alignment accepts translation/similarity/perspective, estimation size,
Lanczos/bicubic/bilinear interpolation, brightness normalization, correlation
threshold, reference index, and exclusion policy.

Export jobs accept `image_id`, `dest`, `format`, `bit_depth`, TIFF `compression`,
JPEG `jpeg_quality`, plus optional `float_tiff_dest` and `depth_dest`. `dng` is
rejected unless the source result is `scene_linear_camera_rgb`.

Interactive OpenAPI JSON is available at `/openapi.json` while the server is running.
