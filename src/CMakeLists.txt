target_sources(CGFinProjLib
	PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/post_processing/tone_mapping.cpp"
        
        "${CMAKE_CURRENT_LIST_DIR}/ray_tracing/embree_interface.cpp"
        
        "${CMAKE_CURRENT_LIST_DIR}/rendering/render.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/rendering/reservoir.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/rendering/screen.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/rendering/shading.cpp"
        
        "${CMAKE_CURRENT_LIST_DIR}/scene/light.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/scene/scene.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/scene/texture.cpp"
        
        "${CMAKE_CURRENT_LIST_DIR}/ui/draw.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/ui/ui.cpp"
        
        "${CMAKE_CURRENT_LIST_DIR}/utils/config.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/utils/utils.cpp")
