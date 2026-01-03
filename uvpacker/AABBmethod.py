# Maya tool for packing UV shells around existing shells.
# Pack (layout) UV of multiple objects together without overlapping on existing unselected uv shells
# Maya 2023 and above only
##
# Rév O'Conner - www.revoconner.com
# github.com/revoconner
##
# License:  MIT 

import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.mel as mel
import math
from collections import defaultdict


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def intersects(self, other):
        return not (self.x >= other.x + other.w or 
                    self.x + self.w <= other.x or
                    self.y >= other.y + other.h or 
                    self.y + self.h <= other.y)
    
    def contains(self, other):
        return (self.x <= other.x and 
                self.y <= other.y and
                self.x + self.w >= other.x + other.w and
                self.y + self.h >= other.y + other.h)


class MaxRectsBSSF:
    def __init__(self, width=1.0, height=1.0):
        self.bin_w = None
        self.bin_h = None
        self.free_rects = None
        self.used_rects = None
        
        self.bin_w = width
        self.bin_h = height
        self.free_rects = [Rect(0, 0, width, height)]
        self.used_rects = []
    
    def add_obstacle(self, x, y, w, h):
        obstacle = None
        new_free = None
        splits = None
        
        obstacle = Rect(x, y, w, h)
        new_free = []
        for free in self.free_rects:
            if free.intersects(obstacle):
                splits = self._split_rect(free, obstacle)
                new_free.extend(splits)
            else:
                new_free.append(free)
        self.free_rects = new_free
        self._prune_free_rects()
    
    def _split_rect(self, free, used):
        result = None
        new_rect = None
        
        result = []
        if used.x < free.x + free.w and used.x + used.w > free.x:
            if used.y > free.y and used.y < free.y + free.h:
                new_rect = Rect(free.x, free.y, free.w, used.y - free.y)
                if new_rect.w > 0.001 and new_rect.h > 0.001:
                    result.append(new_rect)
            if used.y + used.h < free.y + free.h:
                new_rect = Rect(free.x, used.y + used.h, free.w, 
                               free.y + free.h - (used.y + used.h))
                if new_rect.w > 0.001 and new_rect.h > 0.001:
                    result.append(new_rect)
        if used.y < free.y + free.h and used.y + used.h > free.y:
            if used.x > free.x and used.x < free.x + free.w:
                new_rect = Rect(free.x, free.y, used.x - free.x, free.h)
                if new_rect.w > 0.001 and new_rect.h > 0.001:
                    result.append(new_rect)
            if used.x + used.w < free.x + free.w:
                new_rect = Rect(used.x + used.w, free.y,
                               free.x + free.w - (used.x + used.w), free.h)
                if new_rect.w > 0.001 and new_rect.h > 0.001:
                    result.append(new_rect)
        return result
    
    def _prune_free_rects(self):
        i = None
        j = None
        
        i = 0
        while i < len(self.free_rects):
            j = i + 1
            while j < len(self.free_rects):
                if self.free_rects[j].contains(self.free_rects[i]):
                    self.free_rects.pop(i)
                    i -= 1
                    break
                if self.free_rects[i].contains(self.free_rects[j]):
                    self.free_rects.pop(j)
                    j -= 1
                j += 1
            i += 1
    
    def find_position_bssf(self, w, h):
        best_score = None
        best_x = None
        best_y = None
        leftover_h = None
        leftover_w = None
        short_side = None
        long_side = None
        
        best_score = float('inf')
        best_x = None
        best_y = None
        for free in self.free_rects:
            if free.w >= w and free.h >= h:
                leftover_h = abs(free.h - h)
                leftover_w = abs(free.w - w)
                short_side = min(leftover_w, leftover_h)
                long_side = max(leftover_w, leftover_h)
                if short_side < best_score or (short_side == best_score and long_side < best_score):
                    best_x = free.x
                    best_y = free.y
                    best_score = short_side
        return best_x, best_y
    
    def place(self, w, h):
        x = None
        y = None
        used = None
        new_free = None
        splits = None
        
        x, y = self.find_position_bssf(w, h)
        if x is None:
            return None, None
        used = Rect(x, y, w, h)
        self.used_rects.append(used)
        new_free = []
        for free in self.free_rects:
            if free.intersects(used):
                splits = self._split_rect(free, used)
                new_free.extend(splits)
            else:
                new_free.append(free)
        self.free_rects = new_free
        self._prune_free_rects()
        return x, y
    
    def occupancy(self):
        used_area = None
        used_area = sum(r.w * r.h for r in self.used_rects)
        return used_area / (self.bin_w * self.bin_h)


def get_shell_data(mesh):
    shells = None
    original_sel = None
    num_faces = None
    parent = None
    mesh_name = None
    face_comp = None
    shell_ids = None
    sid = None

    shells = defaultdict(lambda: {'faces': set()})
    original_sel = cmds.ls(sl=True, fl=True)

    num_faces = cmds.polyEvaluate(mesh, face=True)
    if not num_faces:
        return {}

    parent = cmds.listRelatives(mesh, p=True, f=True)
    mesh_name = parent[0] if parent else mesh

    for face_idx in range(num_faces):
        face_comp = "{}.f[{}]".format(mesh_name, face_idx)
        cmds.select(face_comp, r=True)
        shell_ids = cmds.polyEvaluate(activeUVShells=True)
        if shell_ids:
            sid = shell_ids[0]
            shells[sid]['faces'].add(face_idx)

    if original_sel:
        cmds.select(original_sel, r=True)
    else:
        cmds.select(cl=True)

    return dict(shells)


def calc_3d_area(mesh, face_indices):
    sel = None
    dag = None
    total = None
    face_iter = None
    
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


def get_shell_uv_area(faces):
    # getUVs() returns UVs in winding order required by shoelace formula
    total = None
    mesh_name = None
    face_idx = None
    sel = None
    dag = None
    face_iter = None
    face_set = None
    u_arr = None
    v_arr = None
    n = None
    area = None
    j = None

    total = 0.0
    if not faces:
        return 0.0001

    mesh_name = faces[0].split('.')[0]
    face_set = set()
    for f in faces:
        face_idx = int(f.split('[')[1].rstrip(']'))
        face_set.add(face_idx)

    sel = om2.MSelectionList()
    sel.add(mesh_name)
    dag = sel.getDagPath(0)

    face_iter = om2.MItMeshPolygon(dag)
    while not face_iter.isDone():
        if face_iter.index() in face_set:
            u_arr, v_arr = face_iter.getUVs()
            n = len(u_arr)
            if n >= 3:
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += u_arr[i] * v_arr[j]
                    area -= u_arr[j] * v_arr[i]
                total += abs(area) / 2.0
        face_iter.next()

    return max(total, 0.0001)


def get_face_uv_bbox(faces):
    all_u = None
    all_v = None
    uvs = None
    coords = None
    
    all_u = []
    all_v = []
    for f in faces:
        uvs = cmds.polyListComponentConversion(f, toUV=True)
        if uvs:
            coords = cmds.polyEditUV(cmds.ls(uvs, fl=True), q=True)
            if coords:
                all_u.extend(coords[0::2])
                all_v.extend(coords[1::2])
    if not all_u:
        return None
    return min(all_u), max(all_u), min(all_v), max(all_v)


def get_shell_center(faces):
    bbox = None
    
    bbox = get_face_uv_bbox(faces)
    if not bbox:
        return 0.5, 0.5
    return (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2


def move_shell(faces, offset_u, offset_v):
    uvs = None
    uv_comps = None
    
    uvs = []
    for f in faces:
        uv_comps = cmds.polyListComponentConversion(f, toUV=True)
        if uv_comps:
            uvs.extend(cmds.ls(uv_comps, fl=True))
    uvs = list(set(uvs))
    if uvs:
        cmds.polyEditUV(uvs, u=offset_u, v=offset_v, r=True)


def scale_shell(faces, scale, pivot_u, pivot_v):
    uvs = None
    uv_comps = None
    coords = None
    new_u = None
    new_v = None

    uvs = []
    for f in faces:
        uv_comps = cmds.polyListComponentConversion(f, toUV=True)
        if uv_comps:
            uvs.extend(cmds.ls(uv_comps, fl=True))
    uvs = list(set(uvs))
    for uv in uvs:
        coords = cmds.polyEditUV(uv, q=True)
        if coords:
            new_u = pivot_u + (coords[0] - pivot_u) * scale
            new_v = pivot_v + (coords[1] - pivot_v) * scale
            cmds.polyEditUV(uv, u=new_u - coords[0], v=new_v - coords[1], r=True)


def rotate_shell(faces, angle_degrees, pivot_u, pivot_v):
    # rotate UVs by angle_degrees around pivot point
    uvs = None
    uv_comps = None
    coords = None
    rad = None
    cos_a = None
    sin_a = None
    rel_u = None
    rel_v = None
    new_u = None
    new_v = None

    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    uvs = []
    for f in faces:
        uv_comps = cmds.polyListComponentConversion(f, toUV=True)
        if uv_comps:
            uvs.extend(cmds.ls(uv_comps, fl=True))
    uvs = list(set(uvs))
    for uv in uvs:
        coords = cmds.polyEditUV(uv, q=True)
        if coords:
            rel_u = coords[0] - pivot_u
            rel_v = coords[1] - pivot_v
            new_u = pivot_u + rel_u * cos_a - rel_v * sin_a
            new_v = pivot_v + rel_u * sin_a + rel_v * cos_a
            cmds.polyEditUV(uv, u=new_u - coords[0], v=new_v - coords[1], r=True)


def get_rotated_bbox_size(w, h, angle_degrees):
    # calculate bounding box size after rotating rectangle by angle
    # non-90 degree rotations increase bbox size
    rad = None
    cos_a = None
    sin_a = None

    rad = math.radians(angle_degrees)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))
    return (w * cos_a + h * sin_a, w * sin_a + h * cos_a)


def get_shell_bbox(uvs):
    us = None
    vs = None

    us = [p[0] for p in uvs]
    vs = [p[1] for p in uvs]
    return min(us), max(us), min(vs), max(vs)


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


def try_pack(shells, base_sizes, obstacle_bboxes, scale, shell_padding, margin, rotation_step=0):
    # rotation_step: 0 = no rotation, 90 = try 0/90/180/270, 45 = try 0/45/90/.../315
    bin_size = None
    packer = None
    ox = None
    oy = None
    ow = None
    oh = None
    cx = None
    cy = None
    cw = None
    ch = None
    sized = None
    w = None
    h = None
    placements = None
    x = None
    y = None
    final_x = None
    final_y = None
    angles = None
    best_x = None
    best_y = None
    best_angle = None
    best_score = None
    rot_w = None
    rot_h = None
    leftover_w = None
    leftover_h = None
    score = None

    bin_size = 1.0 - 2 * margin
    if bin_size <= 0:
        return None

    packer = MaxRectsBSSF(bin_size, bin_size)

    for bbox in obstacle_bboxes:
        ox = bbox[0] - margin - shell_padding
        oy = bbox[2] - margin - shell_padding
        ow = (bbox[1] - bbox[0]) + shell_padding * 2
        oh = (bbox[3] - bbox[2]) + shell_padding * 2
        if ox < bin_size and oy < bin_size and ox + ow > 0 and oy + oh > 0:
            cx = max(0, ox)
            cy = max(0, oy)
            cw = min(bin_size, ox + ow) - cx
            ch = min(bin_size, oy + oh) - cy
            if cw > 0 and ch > 0:
                packer.add_obstacle(cx, cy, cw, ch)

    # generate list of angles to try
    if rotation_step > 0:
        angles = list(range(0, 360, rotation_step))
    else:
        angles = [0]

    sized = []
    for i, shell in enumerate(shells):
        w = base_sizes[i][0] * scale
        h = base_sizes[i][1] * scale
        sized.append((shell, w, h, w * h, i))
    sized.sort(key=lambda x: x[3], reverse=True)

    placements = []
    for shell, w, h, _, orig_idx in sized:
        best_x = None
        best_y = None
        best_angle = 0
        best_score = float('inf')

        # try each rotation angle and find best placement
        for angle in angles:
            if angle == 0:
                rot_w, rot_h = w, h
            else:
                rot_w, rot_h = get_rotated_bbox_size(w, h, angle)

            x, y = packer.find_position_bssf(rot_w, rot_h)
            if x is not None:
                # score based on BSSF - lower is better
                for free in packer.free_rects:
                    if free.w >= rot_w and free.h >= rot_h:
                        leftover_w = abs(free.w - rot_w)
                        leftover_h = abs(free.h - rot_h)
                        score = min(leftover_w, leftover_h)
                        if score < best_score:
                            best_score = score
                            best_x = x
                            best_y = y
                            best_angle = angle
                        break

        if best_x is None:
            return None

        # place at best position with best angle's bbox size
        if best_angle == 0:
            rot_w, rot_h = w, h
        else:
            rot_w, rot_h = get_rotated_bbox_size(w, h, best_angle)
        packer.place(rot_w, rot_h)

        final_x = best_x + margin + shell_padding
        final_y = best_y + margin + shell_padding
        placements.append((shell, final_x, final_y, best_angle))

    return placements, packer.occupancy()


def layout_uvs_maxrects(shell_padding=20, margin=20, min_scale=0.1, binary_iterations=6, debug=False, tile_u=0, tile_v=0, rotation_step=0):
    sel = None
    face_sel = None
    uv_sel = None
    converted = None
    selected_faces_by_mesh = None
    obj = None
    shapes = None
    mesh = None
    face_str = None
    start = None
    end = None
    idx = None
    shells_to_pack = None
    obstacle_bboxes = None
    shell_data = None
    parent = None
    mesh_name = None
    selected_shell_ids = None
    sid = None
    sdata = None
    uvs = None
    bbox = None
    face_comps = None
    area_3d = None
    cx = None
    cy = None
    total_3d = None
    total_uv = None
    target_density = None
    current_uv_area = None
    current_density = None
    scale = None
    base_sizes = None
    scale_high = None
    scale_low = None
    result = None
    test_scale = None
    r = None
    mid = None
    final_scale = None
    placements = None
    occupancy = None
    target_x = None
    target_y = None
    target_angle = None
    offset_u = None
    offset_v = None
    all_meshes = None
    staging_u = None
    staging_v = None
    filtered_obstacles = None

    # staging area is tile_u+2, tile_v+2 to avoid collision with target tile
    staging_u = float(tile_u + 2)
    staging_v = float(tile_v + 2)

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

    # collect meshes from selection and UV editor highlighted
    uv_editor_meshes = get_uv_editor_meshes()
    all_meshes = set(selected_faces_by_mesh.keys())
    all_meshes.update(uv_editor_meshes)
    all_meshes = list(all_meshes)
    if debug:
        print("DEBUG: Meshes in UV editor: {}".format(len(uv_editor_meshes)))

    for mesh in all_meshes:
        cmds.dgdirty(mesh, allPlugs=True)
    cmds.refresh(force=True)

    shells_to_pack = []
    obstacle_bboxes = []
    original_sel = None
    shell_count = None
    shell_sel = None
    uv_sel_shell = None
    coords = None

    original_sel = cmds.ls(sl=True, fl=True)

    for mesh in all_meshes:
        sel_faces = selected_faces_by_mesh.get(mesh, set())
        parent = cmds.listRelatives(mesh, p=True, f=True)
        mesh_name = parent[0] if parent else mesh

        # fast path for meshes with no UV selection - all shells become obstacles
        # uses OpenMaya2 getUvShellsIds() and getUVs() to get all shell bboxes in one pass
        if not sel_faces:
            if debug:
                print("DEBUG: Fast path for {} (all shells -> obstacles)".format(
                    mesh_name.split('|')[-1]))
            try:
                om_sel = om2.MSelectionList()
                om_sel.add(mesh)
                om_dag = om_sel.getDagPath(0)
                om_fn = om2.MFnMesh(om_dag)
                # getUvShellsIds returns (count, array) where array[uv_idx] = shell_id for that UV
                num_shells, shell_ids = om_fn.getUvShellsIds()
                u_arr, v_arr = om_fn.getUVs()
                # group UV coordinates by shell ID to compute per-shell bounding boxes
                shell_uvs = defaultdict(lambda: {'u': [], 'v': []})
                for uv_idx, sid in enumerate(shell_ids):
                    shell_uvs[sid]['u'].append(u_arr[uv_idx])
                    shell_uvs[sid]['v'].append(v_arr[uv_idx])
                for sid, uv_data in shell_uvs.items():
                    if uv_data['u']:
                        bbox_cmds = (min(uv_data['u']), max(uv_data['u']),
                                    min(uv_data['v']), max(uv_data['v']))
                        obstacle_bboxes.append(bbox_cmds)
                        if debug:
                            print("DEBUG:   Shell {} -> obstacle".format(sid))
            except Exception as e:
                if debug:
                    print("DEBUG: Fast path failed: {}".format(e))
            continue

        # normal path: mesh has selection
        shell_data = get_shell_data(mesh)
        if not shell_data:
            if debug:
                print("DEBUG: No shell data for mesh: {}".format(mesh))
            continue
        if debug:
            print("DEBUG: Processing mesh: {} ({} shells found)".format(mesh_name, len(shell_data)))
        selected_shell_ids = set()
        for sid, sdata in shell_data.items():
            if sdata['faces'] & sel_faces:
                selected_shell_ids.add(sid)
        if debug:
            print("DEBUG: Selected shell IDs: {}".format(selected_shell_ids))
        for sid, sdata in shell_data.items():
            if not sdata['faces']:
                continue
            face_comps = ["{}.f[{}]".format(mesh_name, f) for f in sdata['faces']]
            bbox_cmds = get_face_uv_bbox(face_comps)
            if debug:
                print("DEBUG: Shell {} - faces: {}, bbox: {}".format(
                    sid, len(sdata['faces']),
                    "({:.3f},{:.3f})-({:.3f},{:.3f})".format(*bbox_cmds) if bbox_cmds else None))
            if not bbox_cmds:
                continue
            if sid in selected_shell_ids:
                area_3d = calc_3d_area(mesh, sdata['faces'])
                shells_to_pack.append({
                    'faces': face_comps,
                    'mesh': mesh,
                    'area_3d': area_3d
                })
                if debug:
                    print("DEBUG:   -> Added to shells_to_pack")
            else:
                obstacle_bboxes.append(bbox_cmds)
                if debug:
                    print("DEBUG:   -> Added to obstacles")

    # restore original selection
    if original_sel:
        cmds.select(original_sel, r=True)
    else:
        cmds.select(cl=True)

    if not shells_to_pack:
        cmds.warning("No shells to pack")
        return

    # filter obstacles to only those within the target tile (tile_u to tile_u+1, tile_v to tile_v+1)
    # then offset coordinates to be relative to tile origin (0-1 range for packer)
    filtered_obstacles = []
    for obs in obstacle_bboxes:
        # check if obstacle intersects with target tile
        if obs[1] > tile_u and obs[0] < tile_u + 1 and obs[3] > tile_v and obs[2] < tile_v + 1:
            # offset to tile-relative coordinates
            rel_bbox = (obs[0] - tile_u, obs[1] - tile_u, obs[2] - tile_v, obs[3] - tile_v)
            filtered_obstacles.append(rel_bbox)

    if debug:
        print("\nDEBUG SUMMARY:")
        print("  Shells to pack: {}".format(len(shells_to_pack)))
        print("  Total obstacles: {}, in target tile: {}".format(len(obstacle_bboxes), len(filtered_obstacles)))
        for i, obs in enumerate(filtered_obstacles):
            in_bounds = 0 <= obs[0] <= 1 and 0 <= obs[1] <= 1 and 0 <= obs[2] <= 1 and 0 <= obs[3] <= 1
            print("    Obstacle {}: ({:.3f},{:.3f})-({:.3f},{:.3f}) {}".format(
                i, obs[0], obs[2], obs[1], obs[3],
                "" if in_bounds else "<-- OUTSIDE 0-1 RANGE!"))
    
    if debug:
        print("\nDEBUG: Shell positions BEFORE staging:")
        for i, shell in enumerate(shells_to_pack):
            bbox = get_face_uv_bbox(shell['faces'])
            print("  Shell {}: ({:.3f},{:.3f})-({:.3f},{:.3f})".format(i, *bbox) if bbox else "  Shell {}: None".format(i))

    if debug:
        print("\nMoving to staging area ({}, {})...".format(staging_u, staging_v))
    for shell in shells_to_pack:
        cx, cy = get_shell_center(shell['faces'])
        if debug:
            print("  Shell center: ({:.3f},{:.3f}) -> moving by ({:.3f},{:.3f})".format(
                cx, cy, staging_u - cx, staging_v - cy))
        move_shell(shell['faces'], staging_u - cx, staging_v - cy)

    if debug:
        print("\nDEBUG: Shell positions AFTER staging:")
        for i, shell in enumerate(shells_to_pack):
            bbox = get_face_uv_bbox(shell['faces'])
            print("  Shell {}: ({:.3f},{:.3f})-({:.3f},{:.3f})".format(i, *bbox) if bbox else "  Shell {}: None".format(i))

    total_3d = sum(s['area_3d'] for s in shells_to_pack)
    total_uv = sum(get_shell_uv_area(s['faces']) for s in shells_to_pack)
    target_density = total_3d / total_uv if total_uv > 0 else 1.0
    
    for shell in shells_to_pack:
        current_uv_area = get_shell_uv_area(shell['faces'])
        if current_uv_area > 0 and shell['area_3d'] > 0:
            current_density = shell['area_3d'] / current_uv_area
            scale = (current_density / target_density) ** 0.5
            cx, cy = get_shell_center(shell['faces'])
            scale_shell(shell['faces'], scale, cx, cy)
    
    base_sizes = []
    for shell in shells_to_pack:
        bbox = get_face_uv_bbox(shell['faces'])
        if bbox:
            base_sizes.append((bbox[1] - bbox[0] + shell_padding * 2, 
                              bbox[3] - bbox[2] + shell_padding * 2))
        else:
            base_sizes.append((0.1, 0.1))
    
    scale_high = 1.0
    scale_low = None
    
    if debug:
        print("\n=== Phase 1: Find working scale ===")
    
    result = try_pack(shells_to_pack, base_sizes, filtered_obstacles, scale_high, shell_padding, margin, rotation_step)
    if result:
        scale_low = scale_high
        if debug:
            print("Scale 1.0 fits, trying larger...")
        test_scale = 1.5
        while test_scale < 3.0:
            r = try_pack(shells_to_pack, base_sizes, filtered_obstacles, test_scale, shell_padding, margin, rotation_step)
            if r:
                scale_low = test_scale
                test_scale *= 1.2
                if debug:
                    print("Scale {:.3f} fits".format(scale_low))
            else:
                scale_high = test_scale
                if debug:
                    print("Scale {:.3f} fails".format(test_scale))
                break
    else:
        test_scale = scale_high
        while test_scale > min_scale:
            test_scale *= 0.9
            result = try_pack(shells_to_pack, base_sizes, filtered_obstacles, test_scale, shell_padding, margin, rotation_step)
            if result:
                scale_low = test_scale
                if debug:
                    print("Scale {:.3f} fits".format(scale_low))
                break
            else:
                scale_high = test_scale
                if debug:
                    print("Scale {:.3f} fails".format(test_scale))
    
    if scale_low is None:
        print("\nCannot fit at minimum scale")
        return
    
    if debug:
        print("\n=== Phase 2: Binary search [{:.3f}, {:.3f}] ===".format(scale_low, scale_high))
    
    for i in range(binary_iterations):
        mid = (scale_low + scale_high) / 2
        result = try_pack(shells_to_pack, base_sizes, filtered_obstacles, mid, shell_padding, margin, rotation_step)
        if result:
            scale_low = mid
            if debug:
                print("Iter {}: {:.4f} fits".format(i + 1, mid))
        else:
            scale_high = mid
            if debug:
                print("Iter {}: {:.4f} fails".format(i + 1, mid))

    final_scale = scale_low
    result = try_pack(shells_to_pack, base_sizes, filtered_obstacles, final_scale, shell_padding, margin, rotation_step)
    
    if not result:
        print("\nFinal pack failed unexpectedly")
        return
    
    placements, occupancy = result
    
    for shell in shells_to_pack:
        cx, cy = get_shell_center(shell['faces'])
        scale_shell(shell['faces'], final_scale, cx, cy)

    # apply rotations and move to final positions
    rotated_count = 0
    for shell, target_x, target_y, target_angle in placements:
        # apply rotation if needed
        if target_angle != 0:
            cx, cy = get_shell_center(shell['faces'])
            rotate_shell(shell['faces'], target_angle, cx, cy)
            rotated_count += 1
        # move to target position in tile
        bbox = get_face_uv_bbox(shell['faces'])
        if bbox:
            offset_u = (target_x + tile_u) - bbox[0]
            offset_v = (target_y + tile_v) - bbox[2]
            move_shell(shell['faces'], offset_u, offset_v)

    rot_msg = ""
    if rotation_step > 0:
        rot_msg = ", {} rotated".format(rotated_count)
    print("\nSuccess at scale {:.3f}, occupancy {:.1f}% (tile {},{}){}".format(
        final_scale, occupancy * 100, tile_u, tile_v, rot_msg))


class UVLayoutUI:
    WINDOW_NAME = "uvLayoutMultiShellFreeze"
    WINDOW_TITLE = "UV Layout - Pack multi object UV"
    TEXTURE_SIZES = [256, 512, 1024, 2048, 4096, 8192]

    def __init__(self):
        self.widgets = {}
        self.create_ui()
    
    def create_ui(self):
        win = None
        label_width = None
        field_width = None
        row_height = None
        size = None

        if cmds.window(self.WINDOW_NAME, exists=True):
            cmds.deleteUI(self.WINDOW_NAME)

        win = cmds.window(self.WINDOW_NAME, title=self.WINDOW_TITLE, widthHeight=(500, 400), sizeable=True)

        cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnAttach=('both', 10))
        cmds.separator(height=10, style='none')

        label_width = 220
        field_width = 200
        row_height = 24

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
        cmds.text(label="Min Scale (give up below):")
        self.widgets['min_scale'] = cmds.floatField(value=0.1, precision=3, minValue=0.001, maxValue=1.0, width=field_width)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Iterations:")
        self.widgets['iterations'] = cmds.intField(value=10, minValue=1, maxValue=100, width=field_width)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Print Output (for debugging):")
        self.widgets['debug'] = cmds.checkBox(label="", value=False)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="[Texel Density Unify] Preserve 3D ratio:")
        self.widgets['prescale'] = cmds.checkBox(label="", value=True)
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

        cmds.rowLayout(numberOfColumns=2, columnWidth2=(label_width, field_width),
                       columnAlign2=('right', 'left'), columnAttach2=('right', 'left'),
                       columnOffset2=(0, 5), height=row_height)
        cmds.text(label="Rotation Step (0=off, 90=fast):")
        self.widgets['rotation_step'] = cmds.intField(value=0, minValue=0, maxValue=180, width=field_width)
        cmds.setParent('..')

        cmds.text(label="(Non-90 degree rotations increase bbox size - may reduce packing efficiency)",
                  align='right', font='smallObliqueLabelFont')

        cmds.separator(height=6, style='in')

        # center buttons using a nested columnLayout
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

        # collapsible instructions section with resize callbacks
        instructions = (
            "USAGE:\n"
            "  1. Open UV Editor with your mesh(es)\n"
            "  2. Select UV shells you want to pack\n"
            "     (unselected shells become obstacles)\n"
            "  3. Click 'Layout UVs'\n"
            "\n"
            "Objects visible in UV Editor are auto-detected."
        )
        cmds.frameLayout(label="Instructions", collapsable=True, collapse=True,
                         marginWidth=5, marginHeight=5,
                         collapseCommand=self._on_collapse, expandCommand=self._on_expand)
        cmds.scrollField(text=instructions, editable=False, wordWrap=True, height=120, width=460)
        cmds.setParent('..')

        cmds.separator(height=10, style='none')

        cmds.showWindow(win)

    def _on_collapse(self):
        cmds.window(self.WINDOW_NAME, e=True, height=400)

    def _on_expand(self):
        cmds.window(self.WINDOW_NAME, e=True, height=550)

    def execute(self, *args):
        tex_size_label = None
        texture_size = None
        shell_padding_px = None
        margin_px = None
        min_scale = None
        iterations = None
        debug = None
        prescale = None
        tile_u = None
        tile_v = None
        rotation_step = None
        shell_padding = None
        margin = None
        currently_selected_shells = None
        preserve3d = None

        tex_size_label = cmds.optionMenu(self.widgets['texture_size'], q=True, value=True)
        texture_size = int(tex_size_label)

        shell_padding_px = cmds.intField(self.widgets['shell_padding'], q=True, value=True)
        margin_px = cmds.intField(self.widgets['margin'], q=True, value=True)
        min_scale = cmds.floatField(self.widgets['min_scale'], q=True, value=True)
        iterations = cmds.intField(self.widgets['iterations'], q=True, value=True)
        debug = cmds.checkBox(self.widgets['debug'], q=True, value=True)
        prescale = cmds.checkBox(self.widgets['prescale'], q=True, value=True)
        tile_u = cmds.intField(self.widgets['tile_u'], q=True, value=True)
        tile_v = cmds.intField(self.widgets['tile_v'], q=True, value=True)
        rotation_step = cmds.intField(self.widgets['rotation_step'], q=True, value=True)

        shell_padding = float(shell_padding_px) / texture_size
        margin = float(margin_px) / texture_size

        if debug:
            print("Texture: {}px, Padding: {}px ({:.4f} UV), Margin: {}px ({:.4f} UV)".format(
                texture_size, shell_padding_px, shell_padding, margin_px, margin))
            print("Packing to tile: U={}, V={}".format(tile_u, tile_v))
            if rotation_step > 0:
                print("Rotation enabled: step={}deg ({} angles)".format(rotation_step, 360 // rotation_step))

        # prescale shells to unify texel density based on 3D area ratios using u3dLayout
        if prescale:
            currently_selected_shells = " ".join(cmds.ls(sl=True))
            preserve3d = " -res {} -mut 1 -scl 1 -trs 0 -spc 0 -mar 0 -ta 1 -box {} {} {} {} {}".format(
                texture_size, tile_u, tile_u + 1, tile_v, tile_v + 1, currently_selected_shells)
            mel.eval("u3dLayout{}".format(preserve3d))
            if debug:
                print("Prescale: u3dLayout applied with res={}".format(texture_size))

        layout_uvs_maxrects(
            shell_padding=shell_padding,
            margin=margin,
            min_scale=min_scale,
            binary_iterations=iterations,
            debug=debug,
            tile_u=tile_u,
            tile_v=tile_v,
            rotation_step=rotation_step
        )
    
    def close(self, *args):
        if cmds.window(self.WINDOW_NAME, exists=True):
            cmds.deleteUI(self.WINDOW_NAME)


def show_ui():
    ui = None
    ui = UVLayoutUI()
    return ui


if __name__ == "__main__":
    show_ui()