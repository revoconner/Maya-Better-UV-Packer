# Maya UV Layout - Native Bitmap Packer
# Pack UV shells around existing (obstacle) shells using concave bitmap packing
# Maya 2023 and above only
##
# Rév O'Conner - www.revoconner.com
# github.com/revoconner
##
# License:  MIT 


import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.mel as mel
from collections import defaultdict
import subprocess
import tempfile
import os
import time

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = os.path.join(PACKAGE_DIR, "uv_packer.exe")



__all__ = ['show_ui', 'layout_uvs_native', 'BINARY_PATH', 'PACKAGE_DIR']

from . import AABBmethod


def get_shell_data_om2(mesh):
    sel = om2.MSelectionList()
    sel.add(mesh)
    dag = sel.getDagPath(0)
    fn = om2.MFnMesh(dag)
    
    num_shells, shell_ids = fn.getUvShellsIds()
    
    shells = defaultdict(set)
    face_iter = om2.MItMeshPolygon(dag)
    while not face_iter.isDone():
        try:
            uv_idx = face_iter.getUVIndex(0)
            sid = shell_ids[uv_idx]
            shells[sid].add(face_iter.index())
        except:
            pass
        face_iter.next()
    
    return dict(shells), fn, shell_ids


def get_shell_triangles(mesh, face_indices):
    sel = om2.MSelectionList()
    sel.add(mesh)
    dag = sel.getDagPath(0)
    fn = om2.MFnMesh(dag)
    
    u_arr, v_arr = fn.getUVs()
    triangles = []
    
    face_iter = om2.MItMeshPolygon(dag)
    while not face_iter.isDone():
        if face_iter.index() in face_indices:
            try:
                local_verts = list(face_iter.getVertices())
                _, tri_verts = face_iter.getTriangles()
                
                for tri_idx in range(len(tri_verts) // 3):
                    tri_uvs = []
                    for j in range(3):
                        global_vert = tri_verts[tri_idx * 3 + j]
                        if global_vert in local_verts:
                            local_idx = local_verts.index(global_vert)
                            uv_idx = face_iter.getUVIndex(local_idx)
                            tri_uvs.append((u_arr[uv_idx], v_arr[uv_idx]))
                    if len(tri_uvs) == 3:
                        triangles.append(tri_uvs)
            except:
                pass
        face_iter.next()
    
    return triangles


def get_shell_uv_indices(mesh, face_indices):
    sel = om2.MSelectionList()
    sel.add(mesh)
    dag = sel.getDagPath(0)
    
    uv_indices = set()
    face_iter = om2.MItMeshPolygon(dag)
    while not face_iter.isDone():
        if face_iter.index() in face_indices:
            try:
                for i in range(face_iter.polygonVertexCount()):
                    uv_idx = face_iter.getUVIndex(i)
                    uv_indices.add(uv_idx)
            except:
                pass
        face_iter.next()
    
    return list(uv_indices)


def get_shell_3d_area(mesh, face_indices):
    sel = om2.MSelectionList()
    sel.add(mesh)
    dag = sel.getDagPath(0)
    
    total = 0.0
    face_iter = om2.MItMeshPolygon(dag)
    while not face_iter.isDone():
        if face_iter.index() in face_indices:
            total += face_iter.getArea()
        face_iter.next()
    return total


def get_shell_uv_area(mesh, face_indices):
    sel = om2.MSelectionList()
    sel.add(mesh)
    dag = sel.getDagPath(0)
    
    total = 0.0
    face_iter = om2.MItMeshPolygon(dag)
    while not face_iter.isDone():
        if face_iter.index() in face_indices:
            try:
                u_arr, v_arr = face_iter.getUVs()
                n = len(u_arr)
                if n >= 3:
                    area = 0.0
                    for i in range(n):
                        j = (i + 1) % n
                        area += u_arr[i] * v_arr[j] - u_arr[j] * v_arr[i]
                    total += abs(area) / 2.0
            except:
                pass
        face_iter.next()
    return max(total, 0.0001)


def get_shell_bbox(triangles):
    if not triangles:
        return None
    all_u = [p[0] for tri in triangles for p in tri]
    all_v = [p[1] for tri in triangles for p in tri]
    return (min(all_u), max(all_u), min(all_v), max(all_v))


def get_shell_center(triangles):
    bbox = get_shell_bbox(triangles)
    if not bbox:
        return 0.5, 0.5
    return (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2


def move_shell_uvs(mesh, uv_indices, offset_u, offset_v):
    parent = cmds.listRelatives(mesh, p=True, f=True)
    mesh_name = parent[0] if parent else mesh
    uv_comps = ["{}.map[{}]".format(mesh_name, idx) for idx in uv_indices]
    if uv_comps:
        cmds.polyEditUV(uv_comps, u=offset_u, v=offset_v, r=True)


def scale_shell_uvs(mesh, uv_indices, scale, pivot_u, pivot_v):
    parent = cmds.listRelatives(mesh, p=True, f=True)
    mesh_name = parent[0] if parent else mesh
    
    for idx in uv_indices:
        uv = "{}.map[{}]".format(mesh_name, idx)
        coords = cmds.polyEditUV(uv, q=True)
        if coords:
            new_u = pivot_u + (coords[0] - pivot_u) * scale
            new_v = pivot_v + (coords[1] - pivot_v) * scale
            cmds.polyEditUV(uv, u=new_u - coords[0], v=new_v - coords[1], r=True)


def export_packing_data(shells_to_pack, obstacles, margin, padding, texture_size, filepath):
    with open(filepath, 'w') as f:
        f.write("PARAMS\n")
        f.write("margin {}\n".format(margin))
        f.write("padding {}\n".format(padding))
        f.write("texture_size {}\n".format(texture_size))
        f.write("\n")
        
        f.write("SHELLS_TO_PACK\n")
        for i, shell in enumerate(shells_to_pack):
            f.write("SHELL {}\n".format(i))
            for tri in shell['triangles']:
                f.write("TRI {} {} {} {} {} {}\n".format(
                    tri[0][0], tri[0][1],
                    tri[1][0], tri[1][1],
                    tri[2][0], tri[2][1]
                ))
            f.write("UV_INDICES {}\n".format(" ".join(str(x) for x in shell['uv_indices'])))
        f.write("\n")
        
        f.write("OBSTACLES\n")
        for i, obs in enumerate(obstacles):
            f.write("SHELL {}\n".format(i))
            for tri in obs['triangles']:
                f.write("TRI {} {} {} {} {} {}\n".format(
                    tri[0][0], tri[0][1],
                    tri[1][0], tri[1][1],
                    tri[2][0], tri[2][1]
                ))
        f.write("\n")
        f.write("END\n")


def parse_packer_results(filepath):
    results = {}
    with open(filepath, 'r') as f:
        in_results = False
        for line in f:
            line = line.strip()
            if line == "RESULTS":
                in_results = True
            elif line == "END":
                break
            elif in_results and line.startswith("SHELL"):
                parts = line.split()
                shell_id = int(parts[1])
                offset_u = float(parts[2])
                offset_v = float(parts[3])
                rotation = float(parts[4]) if len(parts) > 4 else 0.0
                results[shell_id] = (offset_u, offset_v, rotation)
    return results


def run_native_packer(shells_to_pack, obstacles, margin, padding, texture_size, binary_path, debug=False):
    if not binary_path or not os.path.exists(binary_path):
        cmds.warning("UV packer binary not found: {}".format(binary_path))
        return None
    
    input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    input_path = input_file.name
    output_path = output_file.name
    input_file.close()
    output_file.close()
    
    try:
        export_packing_data(shells_to_pack, obstacles, margin, padding, texture_size, input_path)
        
        if debug:
            print("Running native packer: {}".format(binary_path))
        
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
        with open(output_path, 'w') as out_f:
            result = subprocess.run(
                [binary_path, input_path],
                stdout=out_f,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                timeout=120
            )
        
        if debug and result.stderr:
            for line in result.stderr.decode().strip().split('\n'):
                print("  [packer] {}".format(line))
        
        if result.returncode != 0:
            cmds.warning("Native packer failed with code {}".format(result.returncode))
            return None
        
        return parse_packer_results(output_path)
        
    finally:
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass


def get_uv_editor_meshes():
    highlighted = cmds.ls(hl=True, long=True) or []
    meshes = []
    for obj in highlighted:
        if cmds.objectType(obj) == 'mesh':
            meshes.append(obj)
        elif cmds.objectType(obj) == 'transform':
            shapes = cmds.listRelatives(obj, s=True, f=True, type='mesh')
            if shapes:
                meshes.extend(shapes)
    return list(set(meshes))


def layout_uvs_native(
    shell_padding=20,
    margin=20,
    texture_size=4096,
    tile_u=0,
    tile_v=0,
    rotation_step=0,
    prescale=False,
    debug=False,
    binary_path=None
):
    if binary_path is None:
        binary_path = BINARY_PATH
    
    padding_uv = float(shell_padding) / texture_size
    margin_uv = float(margin) / texture_size
    
    cmds.refresh()
    
    sel = cmds.ls(sl=True, fl=True)
    if not sel:
        cmds.warning("Nothing selected")
        return
    
    face_sel = cmds.filterExpand(sel, sm=34) or []
    uv_sel = cmds.filterExpand(sel, sm=35) or []
    if uv_sel:
        converted = cmds.polyListComponentConversion(uv_sel, toFace=True)
        if converted:
            face_sel.extend(cmds.ls(converted, fl=True))
    if not face_sel:
        cmds.warning("No faces selected")
        return
    face_sel = list(set(face_sel))
    
    selected_faces_str = " ".join(face_sel)
    
    if prescale:
        if debug:
            print("Running u3dLayout prescale...")
        cmd = "u3dLayout -res {} -mut 1 -scl 1 -trs 0 -spc 0 -mar 0 -box {} {} {} {} {}".format(
            texture_size, tile_u, tile_u + 1, tile_v, tile_v + 1, selected_faces_str)
        mel.eval(cmd)
    
    if rotation_step > 0:
        if debug:
            print("Running u3dLayout rotation (step={})...".format(rotation_step))
        cmd = "u3dLayout -res {} -rmn 0 -rmx 360 -rst {} -scl 0 -trs 0 -spc {} -mar {} -box {} {} {} {} {}".format(
            texture_size, rotation_step, padding_uv, margin_uv, tile_u, tile_u + 1, tile_v, tile_v + 1, selected_faces_str)
        mel.eval(cmd)
    
    selected_faces_by_mesh = defaultdict(set)
    for f in face_sel:
        obj = f.split('.')[0]
        if cmds.objectType(obj) == 'transform':
            shapes = cmds.listRelatives(obj, s=True, f=True, type='mesh')
            mesh = shapes[0] if shapes else None
        else:
            mesh = obj
        if mesh:
            face_str = f.split('[')[1].rstrip(']')
            if ':' in face_str:
                start, end = face_str.split(':')
                for idx in range(int(start), int(end) + 1):
                    selected_faces_by_mesh[mesh].add(idx)
            else:
                selected_faces_by_mesh[mesh].add(int(face_str))
    
    uv_editor_meshes = get_uv_editor_meshes()
    
    all_meshes = set(selected_faces_by_mesh.keys())
    all_meshes.update(uv_editor_meshes)
    all_meshes = list(all_meshes)
    
    for mesh in all_meshes:
        cmds.dgdirty(mesh, allPlugs=True)
    cmds.refresh(force=True)
    
    if debug:
        print("\n=== Native UV Packer ===")
        print("Texture: {}px, Padding: {}px, Margin: {}px".format(texture_size, shell_padding, margin))
        print("Target tile: ({}, {}), Rotation step: {}".format(tile_u, tile_v, rotation_step))
        print("Meshes in UV editor: {}".format(len(uv_editor_meshes)))
    
    shells_to_pack = []
    obstacles = []
    original_sel = cmds.ls(sl=True, fl=True)
    
    for mesh in all_meshes:
        sel_faces = selected_faces_by_mesh.get(mesh, set())
        parent = cmds.listRelatives(mesh, p=True, f=True)
        mesh_name = parent[0] if parent else mesh
        short_name = mesh_name.split('|')[-1]
        
        if not sel_faces:
            obs_count_before = len(obstacles)
            try:
                shell_data, _, _ = get_shell_data_om2(mesh)
                for sid, face_set in shell_data.items():
                    triangles = get_shell_triangles(mesh, face_set)
                    if not triangles:
                        continue
                    bbox = get_shell_bbox(triangles)
                    if not bbox:
                        continue
                    if bbox[1] > tile_u and bbox[0] < tile_u + 1 and bbox[3] > tile_v and bbox[2] < tile_v + 1:
                        rel_triangles = [
                            [(p[0] - tile_u, p[1] - tile_v) for p in tri]
                            for tri in triangles
                        ]
                        obstacles.append({'triangles': rel_triangles})
            except Exception as e:
                if debug:
                    print("Obstacle collection for {} FAILED: {}".format(short_name, e))
                continue
            obs_added = len(obstacles) - obs_count_before
            if debug:
                print("{}: {} obstacles in tile".format(short_name, obs_added))
            continue
        
        shell_data, fn, shell_ids = get_shell_data_om2(mesh)
        if not shell_data:
            continue
        
        selected_shell_ids = set()
        for sid, face_set in shell_data.items():
            if face_set & sel_faces:
                selected_shell_ids.add(sid)
        
        obs_count_before = len(obstacles)
        pack_count_before = len(shells_to_pack)
        
        for sid, face_set in shell_data.items():
            triangles = get_shell_triangles(mesh, face_set)
            if not triangles:
                continue
            
            bbox = get_shell_bbox(triangles)
            if not bbox:
                continue
            
            if sid in selected_shell_ids:
                uv_indices = get_shell_uv_indices(mesh, face_set)
                area_3d = get_shell_3d_area(mesh, face_set)
                shells_to_pack.append({
                    'mesh': mesh,
                    'sid': sid,
                    'face_set': face_set,
                    'triangles': triangles,
                    'uv_indices': uv_indices,
                    'area_3d': area_3d
                })
            else:
                if bbox[1] > tile_u and bbox[0] < tile_u + 1 and bbox[3] > tile_v and bbox[2] < tile_v + 1:
                    rel_triangles = [
                        [(p[0] - tile_u, p[1] - tile_v) for p in tri]
                        for tri in triangles
                    ]
                    obstacles.append({'triangles': rel_triangles})
        
        pack_added = len(shells_to_pack) - pack_count_before
        obs_added = len(obstacles) - obs_count_before
        if debug:
            print("{}: {} shells, {} to pack, {} obstacles".format(
                short_name, len(shell_data), pack_added, obs_added))
    
    if original_sel:
        cmds.select(original_sel, r=True)
    else:
        cmds.select(cl=True)
    
    if not shells_to_pack:
        cmds.warning("No shells to pack")
        return
    
    if debug:
        print("\nTotal: {} shells to pack, {} obstacles".format(len(shells_to_pack), len(obstacles)))
    
    if not prescale:
        total_3d = sum(s['area_3d'] for s in shells_to_pack)
        total_uv = sum(get_shell_uv_area(s['mesh'], s['face_set']) for s in shells_to_pack)
        target_density = total_3d / total_uv if total_uv > 0 else 1.0
        
        for shell in shells_to_pack:
            current_uv_area = get_shell_uv_area(shell['mesh'], shell['face_set'])
            if current_uv_area > 0 and shell['area_3d'] > 0:
                current_density = shell['area_3d'] / current_uv_area
                density_scale = (current_density / target_density) ** 0.5
                triangles = get_shell_triangles(shell['mesh'], shell['face_set'])
                cx, cy = get_shell_center(triangles)
                scale_shell_uvs(shell['mesh'], shell['uv_indices'], density_scale, cx, cy)
    
    for shell in shells_to_pack:
        triangles = get_shell_triangles(shell['mesh'], shell['face_set'])
        if not triangles:
            continue
        
        all_u = [p[0] for tri in triangles for p in tri]
        all_v = [p[1] for tri in triangles for p in tri]
        shell['current_min_u'] = min(all_u)
        shell['current_min_v'] = min(all_v)
        
        shell['triangles'] = [
            [(p[0] - shell['current_min_u'], p[1] - shell['current_min_v']) for p in tri]
            for tri in triangles
        ]
    
    start_time = time.time()
    results = run_native_packer(shells_to_pack, obstacles, margin_uv, padding_uv, texture_size, binary_path, debug)
    
    if results is None:
        cmds.warning("Native packer failed - check binary path and console output")
        return
    
    final_scale = 1.0
    if results:
        first_result = list(results.values())[0]
        final_scale = first_result[2]
        if debug:
            print("Final scale from packer: {:.4f}".format(final_scale))
    
    for shell in shells_to_pack:
        triangles = get_shell_triangles(shell['mesh'], shell['face_set'])
        if not triangles:
            continue
        cx, cy = get_shell_center(triangles)
        scale_shell_uvs(shell['mesh'], shell['uv_indices'], final_scale, cx, cy)
    
    for shell_idx, (new_u, new_v, scale_val) in results.items():
        if shell_idx >= len(shells_to_pack):
            continue
        
        shell = shells_to_pack[shell_idx]
        
        current_triangles = get_shell_triangles(shell['mesh'], shell['face_set'])
        bbox = get_shell_bbox(current_triangles)
        if not bbox:
            continue
        
        current_min_u = bbox[0]
        current_min_v = bbox[2]
        
        target_u = new_u + tile_u + padding_uv
        target_v = new_v + tile_v + padding_uv
        
        offset_u = target_u - current_min_u
        offset_v = target_v - current_min_v
        
        move_shell_uvs(shell['mesh'], shell['uv_indices'], offset_u, offset_v)
        
        if debug:
            print("Shell {}: -> ({:.3f},{:.3f})".format(shell_idx, target_u, target_v))
    
    elapsed = time.time() - start_time
    print("\nPacking completed in {:.2f}s (tile {},{})".format(elapsed, tile_u, tile_v))


class UVLayoutUI:
    WINDOW_NAME = "uvLayoutNativePacker"
    WINDOW_TITLE = "UV Layout - Native Bitmap Packer"
    TEXTURE_SIZES = [256, 512, 1024, 2048, 4096, 8192]
    ROTATION_OPTIONS = [("None", 0), ("90 deg", 90), ("45 deg", 45), ("30 deg", 30), ("15 deg", 15)]

    def __init__(self):
        self.widgets = {}
        self.binary_path = BINARY_PATH
        self.create_ui()
    
    def create_ui(self):
        if cmds.window(self.WINDOW_NAME, exists=True):
            cmds.deleteUI(self.WINDOW_NAME)

        win = cmds.window(self.WINDOW_NAME, title=self.WINDOW_TITLE, widthHeight=(500, 480), sizeable=True)
        cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnAttach=('both', 10))
        cmds.separator(height=10, style='none')

        label_width = 220
        field_width = 200
        row_height = 24

        binary_exists = os.path.exists(self.binary_path) if self.binary_path else False
        binary_status = "READY" if binary_exists else "BINARY NOT FOUND"
        status_color = [0.2, 0.6, 0.2] if binary_exists else [0.8, 0.2, 0.2]
        self.widgets['status_text'] = cmds.text(label="Native Packer Status: {}".format(binary_status), 
                                                 backgroundColor=status_color, height=24)
        
        cmds.rowLayout(numberOfColumns=3, columnWidth3=(60, 350, 70),
                       columnAlign3=('left', 'left', 'left'),
                       columnAttach3=('left', 'both', 'right'),
                       columnOffset3=(0, 5, 5), height=row_height)
        cmds.text(label="Binary:")
        self.widgets['binary_path'] = cmds.textField(text=self.binary_path, width=350)
        cmds.button(label="Browse", command=self.browse_binary, width=70)
        cmds.setParent('..')
        
        cmds.separator(height=6, style='in')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Texture Size:")
        self.widgets['texture_size'] = cmds.optionMenu(width=field_width)
        for size in self.TEXTURE_SIZES:
            cmds.menuItem(label=str(size))
        cmds.optionMenu(self.widgets['texture_size'], e=True, select=5)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Shell Padding (px):")
        self.widgets['shell_padding'] = cmds.intField(value=20, minValue=0, maxValue=256, width=field_width)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Margin (px):")
        self.widgets['margin'] = cmds.intField(value=20, minValue=0, maxValue=256, width=field_width)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Rotation (u3dLayout):")
        self.widgets['rotation'] = cmds.optionMenu(width=field_width)
        for label, _ in self.ROTATION_OPTIONS:
            cmds.menuItem(label=label)
        cmds.optionMenu(self.widgets['rotation'], e=True, select=1)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=4, columnWidth4=(label_width, 60, 30, 60),
                       columnAlign4=('right', 'left', 'center', 'left'),
                       columnAttach4=('right', 'left', 'both', 'left'),
                       columnOffset4=(0, 5, 5, 5), height=row_height)
        cmds.text(label="Packing Tile:")
        self.widgets['tile_u'] = cmds.intField(value=0, width=60)
        cmds.text(label="U  V")
        self.widgets['tile_v'] = cmds.intField(value=0, width=60)
        cmds.setParent('..')

        cmds.separator(height=6, style='in')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Print Debug Output:")
        self.widgets['debug'] = cmds.checkBox(label="", value=False)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Prescale Texel Density (u3dLayout):")
        self.widgets['prescale'] = cmds.checkBox(label="", value=True)
        cmds.setParent('..')

        cmds.separator(height=6, style='in')

        cmds.rowLayout(numberOfColumns=3, columnWidth3=(95, 310, 95),
                       columnAlign3=('center', 'center', 'center'),
                       columnAttach3=('both', 'both', 'both'), height=30)
        cmds.text(label="")
        cmds.rowLayout(numberOfColumns=2, columnWidth2=(150, 150),
                       columnAlign2=('center', 'center'), columnAttach2=('both', 'both'),
                       columnOffset2=(0, 10))
        cmds.button(label="Layout UVs", command=self.execute, width=150, height=28)
        cmds.button(label="Close", command=self.close, width=150, height=28)
        cmds.setParent('..')
        cmds.text(label="")
        cmds.setParent('..')

        cmds.separator(height=6, style='in')

        instructions = (
            "USAGE:\n"
            "  1. Open UV Editor with your mesh(es)\n"
            "  2. Select UV shells you want to pack\n"
            "     (unselected shells become obstacles)\n"
            "  3. Click 'Layout UVs'\n"
            "\n"
            "Objects visible in UV Editor are auto-detected.\n"
            "\n"
            "ROTATION:\n"
            "  Uses Maya's u3dLayout for fast rotation\n"
            "  optimization before obstacle-aware packing.\n"
            "  - None: no rotation pre-pass\n"
            "  - 90/45/30/15 deg: rotation step angles"
        )
        cmds.frameLayout(label="Instructions", collapsable=True, collapse=True,
                         marginWidth=5, marginHeight=5,
                         collapseCommand=self._on_collapse, expandCommand=self._on_expand)
        cmds.scrollField(text=instructions, editable=False, wordWrap=True, height=220, width=460)
        cmds.setParent('..')

        cmds.separator(height=10, style='in')

        cmds.text(label="AABB Method: Faster but less tight packing. Uses native Python (no binary required).",
                  align='center', font='smallObliqueLabelFont')
        cmds.button(label="Open AABB Packer", command=self.open_aabb_ui, width=200, height=24)

        cmds.separator(height=10, style='none')
        cmds.showWindow(win)

    def open_aabb_ui(self, *args):
        if cmds.window(self.WINDOW_NAME, exists=True):
            cmds.deleteUI(self.WINDOW_NAME)
        AABBmethod.show_ui()

    def _on_collapse(self):
        cmds.window(self.WINDOW_NAME, e=True, height=480)

    def _on_expand(self):
        cmds.window(self.WINDOW_NAME, e=True, height=730)
    
    def browse_binary(self, *args):
        result = cmds.fileDialog2(
            fileFilter="Executable (*.exe)",
            dialogStyle=2,
            fileMode=1,
            caption="Select UV Packer Binary"
        )
        if result:
            path = result[0]
            cmds.textField(self.widgets['binary_path'], e=True, text=path)
            self.update_status()
    
    def update_status(self):
        path = cmds.textField(self.widgets['binary_path'], q=True, text=True)
        binary_exists = os.path.exists(path) if path else False
        binary_status = "READY" if binary_exists else "BINARY NOT FOUND"
        status_color = [0.2, 0.6, 0.2] if binary_exists else [0.8, 0.2, 0.2]
        cmds.text(self.widgets['status_text'], e=True, 
                  label="Native Packer Status: {}".format(binary_status),
                  backgroundColor=status_color)

    def execute(self, *args):
        binary_path = cmds.textField(self.widgets['binary_path'], q=True, text=True)
        if not binary_path or not os.path.exists(binary_path):
            cmds.warning("UV packer binary not found. Use Browse to locate uv_packer.exe")
            return
        
        texture_size = int(cmds.optionMenu(self.widgets['texture_size'], q=True, value=True))
        shell_padding = cmds.intField(self.widgets['shell_padding'], q=True, value=True)
        margin = cmds.intField(self.widgets['margin'], q=True, value=True)
        debug = cmds.checkBox(self.widgets['debug'], q=True, value=True)
        prescale = cmds.checkBox(self.widgets['prescale'], q=True, value=True)
        tile_u = cmds.intField(self.widgets['tile_u'], q=True, value=True)
        tile_v = cmds.intField(self.widgets['tile_v'], q=True, value=True)
        
        rotation_idx = cmds.optionMenu(self.widgets['rotation'], q=True, select=True) - 1
        rotation_step = self.ROTATION_OPTIONS[rotation_idx][1]

        layout_uvs_native(
            shell_padding=shell_padding,
            margin=margin,
            texture_size=texture_size,
            tile_u=tile_u,
            tile_v=tile_v,
            rotation_step=rotation_step,
            prescale=prescale,
            debug=debug,
            binary_path=binary_path
        )
    
    def close(self, *args):
        if cmds.window(self.WINDOW_NAME, exists=True):
            cmds.deleteUI(self.WINDOW_NAME)


def show_ui():
    return UVLayoutUI()
