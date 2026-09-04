// csv_loader.h — minimal CSV loader (no dependencies).
// Parses a simple comma-separated file. The first line is a header (skipped).
// Returns columns of doubles; the first column (e.g. a date string) is ignored.
#ifndef STATARB_CSV_LOADER_H
#define STATARB_CSV_LOADER_H

#include <cstdio>
#include <string>
#include <vector>

namespace statarb {

// data[i][row] = numeric value of column i (0-based) for row r. Returns false on
// file error or if fewer than `min_cols` numeric columns are found.
inline bool load_csv_columns(const std::string& path, std::vector<std::vector<double>>& out,
                             size_t min_cols = 1) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    out.clear();
    std::string line;
    int c;
    bool header = true;
    std::vector<char> buf;
    while ((c = std::fgetc(f)) != EOF) {
        if (c == '\n') {
            if (header) { header = false; buf.clear(); continue; }   // skip header row
            // parse line
            buf.push_back('\0');
            const char* p = buf.data();
            std::vector<double> row;
            std::string tok;
            int col = 0;
            while (*p) {
                if (*p == ',') {
                    if (col >= 1 && !tok.empty()) row.push_back(std::atof(tok.c_str()));
                    ++col; tok.clear();
                } else {
                    tok.push_back(*p);
                }
                ++p;
            }
            if (col >= 1 && !tok.empty()) row.push_back(std::atof(tok.c_str()));
            if (!row.empty()) {
                if (out.empty()) out.assign(row.size(), std::vector<double>());
                for (size_t i = 0; i < row.size(); ++i) out[i].push_back(row[i]);
            }
            buf.clear();
        } else {
            buf.push_back((char)c);
        }
    }
    std::fclose(f);
    // first column (date string) is not numeric in general; drop it for simplicity
    // by assuming the file has a leading date column we ignore.
    if (out.size() < min_cols) return false;
    return true;
}

}  // namespace statarb
#endif  // STATARB_CSV_LOADER_H
