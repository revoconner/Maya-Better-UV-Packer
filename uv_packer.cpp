// uv_packer.cpp
//  Rév O'Conner
// MIT

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

struct Triangle {
    float u0, v0, u1, v1, u2, v2;
};

struct Shell {
    int id;
    std::vector<Triangle> triangles;
    std::vector<int> uv_indices;
    float min_u, max_u, min_v, max_v;
    float pivot_u, pivot_v;
};

struct PackResult {
    int shell_id;
    float offset_u, offset_v;
    float rotation;
};

// Simple JSON-like parser (minimal, expects specific format)
bool parse_input(std::istream& in, 
                 std::vector<Shell>& shells_to_pack,
                 std::vector<Shell>& obstacles,
                 float& margin,
                 float& padding,
                 int& texture_size) {
    std::string line;
    std::string section;
    Shell current_shell;
    current_shell.id = -1;
    bool in_shell = false;
    bool is_obstacle = false;
    
    while (std::getline(in, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        if (line.empty() || line[0] == '#') continue;
        
        if (line.find("PARAMS") == 0) {
            section = "PARAMS";
        } else if (line.find("SHELLS_TO_PACK") == 0) {
            section = "PACK";
        } else if (line.find("OBSTACLES") == 0) {
            section = "OBSTACLES";
        } else if (line.find("SHELL") == 0) {
            if (in_shell && current_shell.id >= 0) {
                // Save previous shell
                current_shell.min_u = current_shell.max_u = current_shell.triangles[0].u0;
                current_shell.min_v = current_shell.max_v = current_shell.triangles[0].v0;
                for (const auto& tri : current_shell.triangles) {
                    current_shell.min_u = std::min({current_shell.min_u, tri.u0, tri.u1, tri.u2});
                    current_shell.max_u = std::max({current_shell.max_u, tri.u0, tri.u1, tri.u2});
                    current_shell.min_v = std::min({current_shell.min_v, tri.v0, tri.v1, tri.v2});
                    current_shell.max_v = std::max({current_shell.max_v, tri.v0, tri.v1, tri.v2});
                }
                current_shell.pivot_u = (current_shell.min_u + current_shell.max_u) / 2.0f;
                current_shell.pivot_v = (current_shell.min_v + current_shell.max_v) / 2.0f;
                
                if (is_obstacle) {
                    obstacles.push_back(current_shell);
                } else {
                    shells_to_pack.push_back(current_shell);
                }
            }
            
            // Start new shell
            std::istringstream iss(line);
            std::string token;
            iss >> token >> current_shell.id;
            current_shell.triangles.clear();
            current_shell.uv_indices.clear();
            in_shell = true;
            is_obstacle = (section == "OBSTACLES");
            
        } else if (line.find("TRI") == 0) {
            Triangle tri;
            std::istringstream iss(line);
            std::string token;
            iss >> token >> tri.u0 >> tri.v0 >> tri.u1 >> tri.v1 >> tri.u2 >> tri.v2;
            current_shell.triangles.push_back(tri);
            
        } else if (line.find("UV_INDICES") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            int idx;
            while (iss >> idx) {
                current_shell.uv_indices.push_back(idx);
            }
            
        } else if (section == "PARAMS") {
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            if (key == "margin") iss >> margin;
            else if (key == "padding") iss >> padding;
            else if (key == "texture_size") iss >> texture_size;
        } else if (line.find("END") == 0) {
            // Save last shell
            if (in_shell && current_shell.id >= 0 && !current_shell.triangles.empty()) {
                current_shell.min_u = current_shell.max_u = current_shell.triangles[0].u0;
                current_shell.min_v = current_shell.max_v = current_shell.triangles[0].v0;
                for (const auto& tri : current_shell.triangles) {
                    current_shell.min_u = std::min({current_shell.min_u, tri.u0, tri.u1, tri.u2});
                    current_shell.max_u = std::max({current_shell.max_u, tri.u0, tri.u1, tri.u2});
                    current_shell.min_v = std::min({current_shell.min_v, tri.v0, tri.v1, tri.v2});
                    current_shell.max_v = std::max({current_shell.max_v, tri.v0, tri.v1, tri.v2});
                }
                current_shell.pivot_u = (current_shell.min_u + current_shell.max_u) / 2.0f;
                current_shell.pivot_v = (current_shell.min_v + current_shell.max_v) / 2.0f;
                
                if (is_obstacle) {
                    obstacles.push_back(current_shell);
                } else {
                    shells_to_pack.push_back(current_shell);
                }
            }
            break;
        }
    }
    
    return !shells_to_pack.empty();
}

// Custom packer, inspired by xatlas but simpliefied
class SimpleBitmapPacker {
public:
    int radix;
    float scale;
    std::vector<uint8_t> bitmap;
    
    SimpleBitmapPacker(int resolution, float uv_scale) 
        : radix(resolution), scale(uv_scale), bitmap(resolution * resolution, 0) {}
    
    void clear() {
        std::fill(bitmap.begin(), bitmap.end(), 0);
    }
    
    int uv_to_cell(float uv) const {
        return std::max(0, std::min(radix - 1, (int)(uv * radix / scale)));
    }
    
    void rasterize_line(float x0, float y0, float x1, float y1) {
        int ix0 = uv_to_cell(x0), iy0 = uv_to_cell(y0);
        int ix1 = uv_to_cell(x1), iy1 = uv_to_cell(y1);
        
        int dx = std::abs(ix1 - ix0);
        int dy = std::abs(iy1 - iy0);
        int sx = ix0 < ix1 ? 1 : -1;
        int sy = iy0 < iy1 ? 1 : -1;
        int err = dx - dy;
        
        for (int i = 0; i < dx + dy + 1; i++) {
            if (ix0 >= 0 && ix0 < radix && iy0 >= 0 && iy0 < radix) {
                bitmap[iy0 * radix + ix0] = 1;
            }
            if (ix0 == ix1 && iy0 == iy1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; ix0 += sx; }
            if (e2 < dx) { err += dx; iy0 += sy; }
        }
    }
    
    void rasterize_triangle(const Triangle& tri, float offset_u = 0, float offset_v = 0) {
        float u0 = tri.u0 + offset_u, v0 = tri.v0 + offset_v;
        float u1 = tri.u1 + offset_u, v1 = tri.v1 + offset_v;
        float u2 = tri.u2 + offset_u, v2 = tri.v2 + offset_v;
        
        // Rasterize edges
        rasterize_line(u0, v0, u1, v1);
        rasterize_line(u1, v1, u2, v2);
        rasterize_line(u2, v2, u0, v0);
        
        // Fill interior
        int min_x = uv_to_cell(std::min({u0, u1, u2}));
        int max_x = uv_to_cell(std::max({u0, u1, u2}));
        int min_y = uv_to_cell(std::min({v0, v1, v2}));
        int max_y = uv_to_cell(std::max({v0, v1, v2}));
        
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                float cx = (x + 0.5f) * scale / radix;
                float cy = (y + 0.5f) * scale / radix;
                
                // Barycentric test
                auto sign = [](float px, float py, float ax, float ay, float bx, float by) {
                    return (px - bx) * (ay - by) - (ax - bx) * (py - by);
                };
                
                float d1 = sign(cx, cy, u0, v0, u1, v1);
                float d2 = sign(cx, cy, u1, v1, u2, v2);
                float d3 = sign(cx, cy, u2, v2, u0, v0);
                
                bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
                bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
                
                if (!(has_neg && has_pos)) {
                    bitmap[y * radix + x] = 1;
                }
            }
        }
    }
    
    void dilate(int amount) {
        if (amount <= 0) return;
        std::vector<uint8_t> temp = bitmap;
        for (int y = 0; y < radix; y++) {
            for (int x = 0; x < radix; x++) {
                if (temp[y * radix + x]) {
                    for (int dy = -amount; dy <= amount; dy++) {
                        for (int dx = -amount; dx <= amount; dx++) {
                            int nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < radix && ny >= 0 && ny < radix) {
                                bitmap[ny * radix + nx] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    void rasterize_shell(const Shell& shell, float offset_u = 0, float offset_v = 0) {
        for (const auto& tri : shell.triangles) {
            rasterize_triangle(tri, offset_u, offset_v);
        }
    }
    
    bool can_place(const std::vector<uint8_t>& chart, int chart_w, int chart_h, int ox, int oy) {
        if (ox < 0 || oy < 0 || ox + chart_w > radix || oy + chart_h > radix) return false;
        for (int y = 0; y < chart_h; y++) {
            for (int x = 0; x < chart_w; x++) {
                if (chart[y * chart_w + x] && bitmap[(oy + y) * radix + (ox + x)]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    void place(const std::vector<uint8_t>& chart, int chart_w, int chart_h, int ox, int oy) {
        for (int y = 0; y < chart_h; y++) {
            for (int x = 0; x < chart_w; x++) {
                if (chart[y * chart_w + x]) {
                    bitmap[(oy + y) * radix + (ox + x)] = 1;
                }
            }
        }
    }
};

struct ChartData {
    std::vector<uint8_t> bitmap;
    int width, height;
    float min_u, min_v;
    int shell_idx;
    float perimeter;
};

std::vector<PackResult> pack_shells(
    std::vector<Shell>& shells_to_pack,
    const std::vector<Shell>& obstacles,
    float margin,
    float padding,
    int texture_size) {
    
    std::vector<PackResult> results;
    
    // Fixed 1.0 UV space - pack into margin to (1 - margin)
    float bin_size = 1.0f - 2.0f * margin;
    if (bin_size <= 0) {
        std::cerr << "ERROR: Margin too large" << std::endl;
        return results;
    }
    
    int resolution = 1024;
    int padding_cells = std::max(1, (int)std::ceil(padding * resolution / bin_size));
    
    // Calculate base sizes (shell size only, padding handled separately)
    std::vector<float> base_widths, base_heights;
    for (const auto& s : shells_to_pack) {
        float w = s.max_u - s.min_u;
        float h = s.max_v - s.min_v;
        base_widths.push_back(w);
        base_heights.push_back(h);
    }
    
    // Binary search for optimal scale
    float scale_low = 0.01f;
    float scale_high = 2.0f;
    float best_scale = 0.0f;
    std::vector<std::pair<int, std::pair<float, float>>> best_placements;
    
    for (int iter = 0; iter < 20; iter++) {
        float test_scale = (scale_low + scale_high) / 2.0f;
        
        // Create global packer
        SimpleBitmapPacker packer(resolution, bin_size);
        
        // Rasterize obstacles (already in 0-1 tile-relative coords)
        for (const auto& obs : obstacles) {
            packer.rasterize_shell(obs);
        }
        // Dilate obstacles by padding
        packer.dilate(padding_cells);
        
        // Build sorted chart list (larger first for better packing)
        struct ChartInfo {
            int idx;
            float w, h;
            float area;
        };
        std::vector<ChartInfo> charts;
        for (size_t i = 0; i < shells_to_pack.size(); i++) {
            float w = base_widths[i] * test_scale;
            float h = base_heights[i] * test_scale;
            charts.push_back({(int)i, w, h, w * h});
        }
        std::sort(charts.begin(), charts.end(), [](const ChartInfo& a, const ChartInfo& b) {
            return a.area > b.area;
        });
        
        // Try to place all charts
        bool all_placed = true;
        std::vector<std::pair<int, std::pair<float, float>>> placements;
        
        for (const auto& chart : charts) {
            const Shell& shell = shells_to_pack[chart.idx];
            float w = chart.w;
            float h = chart.h;
            
            // Calculate chart bitmap size (with padding)
            float total_w = w + padding * 2;
            float total_h = h + padding * 2;
            int chart_w = std::max(2, (int)std::ceil(total_w * resolution / bin_size) + 1);
            int chart_h = std::max(2, (int)std::ceil(total_h * resolution / bin_size) + 1);
            
            // Rasterize shell into local bitmap
            // Local bitmap covers the shell + padding area
            SimpleBitmapPacker local(std::max(chart_w, chart_h), total_w);
            
            for (const auto& tri : shell.triangles) {
                // Triangles are already normalized to origin by Python
                // Scale them and offset by padding
                Triangle scaled_tri = {
                    tri.u0 * test_scale + padding,
                    tri.v0 * test_scale + padding,
                    tri.u1 * test_scale + padding,
                    tri.v1 * test_scale + padding,
                    tri.u2 * test_scale + padding,
                    tri.v2 * test_scale + padding
                };
                local.rasterize_triangle(scaled_tri);
            }
            
            // Dilate shell by padding (for shell-to-shell spacing)
            int local_padding_cells = std::max(1, (int)std::ceil(padding * local.radix / total_w));
            local.dilate(local_padding_cells);
            
            // Extract chart bitmap
            std::vector<uint8_t> chart_bitmap(chart_w * chart_h, 0);
            for (int y = 0; y < chart_h && y < local.radix; y++) {
                for (int x = 0; x < chart_w && x < local.radix; x++) {
                    chart_bitmap[y * chart_w + x] = local.bitmap[y * local.radix + x];
                }
            }
            
            // Find first valid placement (bottom-left scan)
            bool placed = false;
            for (int y = 0; y <= resolution - chart_h && !placed; y++) {
                for (int x = 0; x <= resolution - chart_w && !placed; x++) {
                    if (packer.can_place(chart_bitmap, chart_w, chart_h, x, y)) {
                        packer.place(chart_bitmap, chart_w, chart_h, x, y);
                        
                        // Convert cell position to UV space
                        // Add margin and account for padding offset in the bitmap
                        float new_u = x * bin_size / resolution + margin;
                        float new_v = y * bin_size / resolution + margin;
                        
                        placements.push_back({chart.idx, {new_u, new_v}});
                        placed = true;
                    }
                }
            }
            
            if (!placed) {
                all_placed = false;
                break;
            }
        }
        
        if (all_placed) {
            scale_low = test_scale;
            best_scale = test_scale;
            best_placements = placements;
        } else {
            scale_high = test_scale;
        }
    }
    
    if (best_scale <= 0) {
        std::cerr << "ERROR: Could not fit shells at minimum scale" << std::endl;
        return results;
    }
    
    std::cerr << "INFO: Final scale = " << best_scale << std::endl;
    
    // Build results
    for (const auto& p : best_placements) {
        PackResult pr;
        pr.shell_id = shells_to_pack[p.first].id;
        pr.offset_u = p.second.first;   // Position includes margin
        pr.offset_v = p.second.second;
        pr.rotation = best_scale;  // Scale stored in rotation field
        results.push_back(pr);
    }
    
    return results;
}

int main(int argc, char* argv[]) {
    std::vector<Shell> shells_to_pack;
    std::vector<Shell> obstacles;
    float margin = 0.005f;
    float padding = 0.005f;
    int texture_size = 4096;
    
    // Read from stdin or file
    std::istream* input = &std::cin;
    std::ifstream file;
    
    if (argc > 1) {
        file.open(argv[1]);
        if (!file) {
            std::cerr << "ERROR: Cannot open file " << argv[1] << std::endl;
            return 1;
        }
        input = &file;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!parse_input(*input, shells_to_pack, obstacles, margin, padding, texture_size)) {
        std::cerr << "ERROR: Failed to parse input or no shells to pack" << std::endl;
        return 1;
    }
    
    std::cerr << "INFO: Packing " << shells_to_pack.size() << " shells with " 
              << obstacles.size() << " obstacles" << std::endl;
    
    auto results = pack_shells(shells_to_pack, obstacles, margin, padding, texture_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Output results
    std::cout << "RESULTS" << std::endl;
    for (const auto& r : results) {
        std::cout << "SHELL " << r.shell_id << " " << r.offset_u << " " << r.offset_v << " " << r.rotation << std::endl;
    }
    std::cout << "END" << std::endl;
    
    std::cerr << "INFO: Completed in " << duration.count() << "ms" << std::endl;
    
    return 0;
}