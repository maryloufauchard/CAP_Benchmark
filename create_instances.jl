
include("algorithms.jl")
include("Instances.jl")
using Random
using ArgParse
 
function create_input_file_scp(
    filename::String,
    num_students::Int,
    num_schools::Int,
    capacities,
    pref_mode,
    seed,
    incomplete_len
)
 
    if length(capacities) != num_schools
        error("The length of capacities must match the number of schools.")
    end
 
    open(filename, "w") do file
        println(file, "# Num. students:$num_students")
        println(file, "# Num. colleges:$num_schools")
 
        students = ["s$i" for i in 1:num_students]
        colleges = ["c$j" for j in 1:num_schools]
        println(file, "# Students:" * join(students, ","))
        println(file, "# Colleges:" * join(colleges, ","))
 
        println(file, "# Capacities:")
        for (i, capacity) in enumerate(capacities)
            println(file, "c$i $capacity")
        end
 
        rng = MersenneTwister(seed)
 
        println(file, "# Student preferences:")
        for student in students
            if pref_mode == "Flexible"
                lenght_pref = rand(rng, incomplete_len:num_schools)
            elseif pref_mode == "Complete"
                lenght_pref = num_schools
            elseif pref_mode == "Incomplete"
                lenght_pref = incomplete_len
            else
                println("INVALID PREFERRENCES")
            end
            sampled_colleges = shuffle(rng, colleges)[1:lenght_pref]
            pref = join(["($(index),$(element))" for (index, element) in enumerate(sampled_colleges)], " ")
            println(file, "$student " * pref)
        end
 
        println(file, "# College priorities:")
        for college in colleges
            sampled_students = shuffle(rng, students)
            pref = join(["($(index),$(element))" for (index, element) in enumerate(sampled_students)], " ")
            println(file, "$college " * pref)
        end
    end
 
    println("File '$filename' created successfully.")
end
 
function create_match_file(
    filename_input::String,
    filename_match::String
)
    open(filename_match, "w") do file
        SC = read_game_from_txt(filename_input)
        DA_match = DA(SC)
        DA_match_tuple = dict_to_tuples(DA_match)
        println(file, "# DA matching of corresponding input")
        println(file, DA_match_tuple)
    end
    println("File '$filename_match' created successfully.")
end
 
function create_equal_capacity(totalCap, schools)
    return fill(div(totalCap, schools), schools)
end
 
function distribute_seats(total_capacity, num_schools, seed)
    if num_schools > total_capacity
        error("Number of schools cannot exceed total capacity.")
    end
 
    rng = MersenneTwister(seed)
 
    # Each school gets at least one seat
    seats = ones(Int, num_schools)
    remaining = total_capacity - num_schools
 
    for _ in 1:remaining
        idx = rand(rng, 1:num_schools)
        seats[idx] += 1
    end
 
    return seats
end
 
function dict_to_tuples(d::Dict{String,String})
    tuples = [(k, v) for (k, v) in d]
    return sort(tuples, by = x -> parse(Int, replace(x[1], "s" => "")))
end
 
##########################################################################################################################
################################################# ARGUMENT PARSING #######################################################
##########################################################################################################################
parse_int_list(s::AbstractString) = parse.(Int, split(s, ","))
parse_str_list(s::AbstractString) = String.(split(s, ","))
 
function parse_commandline()
    s = ArgParseSettings(description = "Generate school-choice-problem (SCP) instances and their DA matchings.")
 
    @add_arg_table! s begin
        "--num-students"
            help = "Comma-separated list of student counts""
            arg_type = String
            default = "5,10,15,20"
        "--preferences"
            help = "Comma-separated list of preference modes: Incomplete, Flexible, Complete"
            arg_type = String
            default = "Incomplete,Flexible,Complete"
        "--seeds"
            help = "Comma-separated list of seeds"
            arg_type = String
            default = "1,2,3"
        "--incomplete-len"
            help = "Length of incomplete/minimum preference lists, normally stays at 1"
            arg_type = Int
            default = 1
        "--input-dir"
            help = "Directory to write instance input files to"
            arg_type = String
            default = "LLM_instances_final"
        "--match-dir"
            help = "Directory to write DA matching files to"
            arg_type = String
            default = "LLM_match_final"
    end
 
    return parse_args(s)
end

# number of schools and total capacity depends on number of students and therefore easier to adjust here to all values that need to be tested
const NUMBER_SCHOOL = Dict(5 => [5, 4, 3], 10 => [10, 5, 4, 3], 15 => [15, 8, 5, 4],
                            20 => [20, 10, 7, 5], 30 => [30, 15, 10, 8])
 
const TOTAL_CAPACITY = Dict(5 => [4, 5, 10], 10 => [8, 10, 15], 15 => [12, 15, 20],
                             20 => [16, 20, 25], 30 => [24, 30, 35])
 
function main()
    args = parse_commandline()
 
    number_students = parse_int_list(args["num-students"])
    preferences = parse_str_list(args["preferences"])
    seeds = parse_int_list(args["seeds"])
    incomplete_len = args["incomplete-len"]
    input_dir = args["input-dir"]
    match_dir = args["match-dir"]
 
    mkpath(input_dir)
    mkpath(match_dir)
 
    for n_student in number_students
        for n_school in NUMBER_SCHOOL[n_student]
            for pref_mode in preferences
                for total_cap in TOTAL_CAPACITY[n_student]
 
                    # if nb_school > total_cap : skip
                    if n_school > total_cap
                        continue
                    end
 
                    for seed in seeds
                        school_capacity = distribute_seats(total_cap, n_school, seed)
                        filename_input = joinpath(input_dir, "scp_($(n_student),$(n_school))_$(pref_mode)_$(total_cap)_seed$(seed).txt")
                        filename_output = joinpath(match_dir, "match_scp_($(n_student),$(n_school))_$(pref_mode)_$(total_cap)_seed$(seed).txt")
 
                        create_input_file_scp(filename_input, n_student, n_school, school_capacity, pref_mode, seed, incomplete_len)
                        create_match_file(filename_input, filename_output)
                    end
                end
            end
        end
    end
end
 
main()
