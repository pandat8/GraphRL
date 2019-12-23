using MatrixDepot

ss = mdlist(sp(:))  # list of all SuiteSparse matrices

ss_small = joinpath(@__DIR__, "ss_small.sh")
ss_large = joinpath(@__DIR__, "ss_large.sh")

open(ss_small, "w") do file end
open(ss_large ,"w") do file end

for (i, mat) in enumerate(ss)
    (i % 100 == 0) && (@info "\t\t$i/$(length(ss))\t\t")
    # load matrix
    # md = mdopen(mat)

    md = mdinfo(mat)
    s = parse.(Int, split(md.content[end].content[1]))
    m, n = s[1], s[2]
    # m, n, nz = s)

    # Dimension check
    m == n || continue

    # dimension filter
    if 50 <= m <= 500
        # small matrix
        @info "Small: $mat"
        open(ss_small, "a") do file
            write(file, "wget https://sparse.tamu.edu/MM/$(mat).tar.gz -P ./ss_small/\n")
        end
    elseif 1000 <= m <= 2000
        # large matrix
        @info "Large: $mat"
        open(ss_large, "a") do file
            write(file, "wget https://sparse.tamu.edu/MM/$(mat).tar.gz -P ./ss_large/\n")
        end
    end
end


