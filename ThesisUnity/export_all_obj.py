import bpy


def export_all_obj(filename, exportFolder):
    bpy.ops.import_scene.obj(filepath = filename)
    objects = bpy.data.objects
    for object in objects:
        print(object)
        bpy.ops.object.select_all(action='DESELECT')
        object.select = True
        exportName = exportFolder + object.name + '.obj'
        bpy.ops.export_scene.obj(filepath=exportName, use_selection=True)


export_all_obj('unityexport.obj', 'export_all\\')