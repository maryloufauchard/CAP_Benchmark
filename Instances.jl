# credits given to  Federico Bobbio, Margarida Carvalho, Andrea Lodi, Ignacio Rios and Alfredo Torrico

struct SchoolChoice
    students::Vector{String} # list of students
    colleges::Vector{String} # list of colleges
    pref::Dict{String, Dict{Int, String}} # dictionary of preferences
    cap::Dict{String,Int} # dictionary of capacities
    numbS::Int # number of students
    numbC::Int # number of schools
    B::Int # budget
    S::Dict{String, Dict{String, Vector{String}}} # S[student][school] set of schools that the student prefers over school+school
    T::Dict{String, Dict{String, Vector{String}}} # T[student][school] set of students that school prefers over student+student
end

# filename="inputs/Margarida/sim=0.txt"
function read_game_from_txt(filename::AbstractString)::SchoolChoice
    file = open(filename, "r")
    aux_cap = false
    aux_pref_students = false
    aux_pref_schools = false
    numbC = -1
    numbS = -1
    B = -1
    i = -1
    students = Vector{String}()
    colleges = Vector{String}()
    cap = Dict{String,Int}()
    pref = Dict{String, Dict{Int, String}}()
    for line in eachline(file)
        # Split the line by ":" and get the second part
        parts = split(line, ":")
        if startswith(parts[1], "# Num. students")
            value = strip(parts[2])
            numbS = parse(Int, value)
        elseif startswith(parts[1], "# Num. colleges")
            value = strip(parts[2])
            numbC = parse(Int, value)
        elseif startswith(parts[1], "# Budget")
            value = strip(parts[2])
            B = parse(Int, value)
        elseif startswith(parts[1], "# Students")
            value = strip(parts[2])
            students = split(value, ",")
        elseif startswith(parts[1], "# Colleges")
            value = strip(parts[2])
            colleges = split(value, ",")
        elseif startswith(parts[1], "# Capacities")
            aux_cap = true
        elseif startswith(parts[1],"# Student preferences")
            aux_pref_students = true
        elseif startswith(parts[1],"# College priorities")
            aux_pref_schools = true
        end

        if aux_cap
            if 1<=i<= numbC
                parts = split(line, " ")
                value = strip(parts[2])
                cap[parts[1]] = parse(Int, value)
                i = i+1
            elseif i > numbC
                aux_cap = false
                i=-1
            else
                i=1
            end
        end

        if aux_pref_students 
            if 1<=i<= numbS
                key = match(r"\w+", line).match
                # Find all tuples in the string
                tuples = eachmatch(r"\((\d+),(\w+)\)", line)
                pref[key] = Dict{Int, String}(parse(Int, m.captures[1]) => m.captures[2] for m in tuples)
                i = i+1
            elseif i > numbS
                aux_pref_students = false
                i=-1
            else
                i=1
            end
        end

        if aux_pref_schools
            if 1<=i<= numbC
                key = match(r"\w+", line).match
                # Find all tuples in the string
                tuples = eachmatch(r"\((\d+),(\w+)\)", line)
                pref[key] = Dict{Int, String}(parse(Int, m.captures[1]) => m.captures[2] for m in tuples)
                i = i+1
            elseif i > numbC
                aux_pref_schools = false
                i=-1
            else
                i=1
            end
        end
    end
    close(file)
    # S[student][school] set of schools that the student prefers over school+school
    S = Dict{String, Dict{String, Vector{String}}}()
    # T[student][school] set of schools that the student prefers over school+school
    T = Dict{String, Dict{String, Vector{String}}}()
    for s in students
        S[s] = Dict{String, Vector{String}}()
        T[s] = Dict{String, Vector{String}}()
        for c in colleges
            i = 1
            while pref[s][i] != c
                i+=1
                if i>length(pref[s])
                    i = length(pref[s])
                    break
                end
            end
            S[s][c] = [pref[s][j] for j in 1:i]
            
            k = 1
            while pref[c][k] != s
                k+=1
                if k>length(pref[c])
                    k = length(pref[c])
                    break
                end
            end
            T[s][c] = [pref[c][j] for j in 1:k]
        end
    end

    return SchoolChoice(students,colleges,pref,cap,numbS,numbC,B,S,T)
end

function write_output(outputs, outfile,timelimit=3600)
    f = open(outfile, "w")
    if timelimit<outputs["runtime"]
        write(f, "# No optimal solution found \n")
        write(f, "# Best primal bound from Gurobi (upper bound):", string(outputs["obj"]), "\n")
        write(f, "# Best dual bound from Gurobi (lower bound):", string(outputs["LB"]), "\n")
    else
        write(f, "# Objective:", string(outputs["obj"]), "\n")
    end
    write(f, "# Num Vars:", "Not computed \n")
    write(f, "# Num Cols:", "Not computed \n")
    # write(f, "# Mean preference of assignment:", outputs["mean_pref"], "\n")
    # write(f, "# Num unassigned:", outputs["num_unassigned"], "\n")
    # write(f, "# Max extra:", outputs["max_extra"], "\n")
    # write(f, "# Additional seats:", outputs["add_seats"], "\n")
    write(f, "# Run time:", string(outputs["runtime"]), "\n")
    write(f, "# MipGap:", string(outputs["mipgap"]), "\n")
    write(f, "# Time in callback:", string(outputs["time callback"]), "\n")
    write(f, "# Num callbacks:", string(outputs["num callback"]), "\n")
    write(f, "# Time of best UB in callbacks:", string(outputs["time best ub"]), "\n")
    write(f, "# Value of best UB in callbacks:", string(outputs["value best ub"]), "\n")

    # Extras
    if haskey(outputs, "nodes")
        write(f, "# Nodes:", string(outputs["nodes"]), "\n")
    end
    if haskey(outputs, "obj_root")
        write(f, "# Objective Root:", outputs["obj_root"], "\n")
    end
    if haskey(outputs, "obj_lr")
        write(f, "# Objective LR:", outputs["obj_lr"], "\n")
    end
    if haskey(outputs, "gap_root")
        write(f, "# Gap Root:", outputs["gap_root"], "\n")
    end
    if haskey(outputs, "gap_lr")
        write(f, "# Gap LR:", outputs["gap_lr"], "\n")
    end

    # write(f, "# Distribution preference of assignment:\n")
    # for p in sort(collect(keys(outputs["pref_assignment"])))
    #     write(f, "$p $(outputs["pref_assignment"][p])\n")
    # end

    write(f, "# Optimal x:\n")
    for id_s in keys(outputs["x_opt"])
        write(f, "$id_s $(outputs["x_opt"][id_s])\n")
        # for id_c in keys(outputs["x_opt"][id_s])
        #     write(f, "$id_s $id_c $(outputs["x_opt"][id_s][id_c])\n")
        # end
    end

    write(f, "# Optimal t:\n")
    for id_c in keys(outputs["t_opt"])
        write(f, "$id_c $(outputs["t_opt"][id_c])\n")
    end

    close(f)
end
