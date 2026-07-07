 # credit given to  given to Federico Bobbio, Margarida Carvalho, Andrea Lodi, Ignacio Rios and Alfredo Torrico under paper Capacity Planning in Stable Matching
using Gurobi
using JuMP

# DA : student-optimal matching
# extra_seats[school] = integer number
function DA(SC,extra_seats=0)
    if extra_seats == 0
        extra_seats=Dict(i => 0 for i in SC.colleges)
    end
    # this stores the current preference of each student
    cp = Dict(i => 1 for i in SC.students)
    matching = Dict(i => "nothing" for i in SC.students)
    is_matched = Dict(i => false for i in SC.students)

    # Each student proposes to its current top school
    proposals = Dict(c => [] for c in SC.colleges)
    rejected = copy(SC.students)
    it = 0
    while true
        it+=1
        # in the first iteration we consider all students; later we only consider those previously rejected
        for m in rejected
            if is_matched[m]
                continue
            end
            try
                push!(proposals[SC.pref[m][cp[m]]],m)
            catch
                println(m," ", cp[m])
                error("An error occured")
            end
        end
        
        rejected = Vector{String}()
        # Each school rejects those student who are not in the top
        for c in SC.colleges
            # if the number of proposals received is less than the vacants
            if length(proposals[c]) <= SC.cap[c]+extra_seats[c]
                continue
            end
            # otherwise, we order the proposals according to school preferences and reject those students who are not in the top
            top = [SC.pref[c][k] for k in sort(collect((keys(SC.pref[c])))) if SC.pref[c][k] in proposals[c]]
            # Reject students who are not in the top preferences
            for i in round(Int,SC.cap[c]+extra_seats[c]+1):length(proposals[c])
                deleteat!(proposals[c], findall(x -> x == top[i], proposals[c]))
                push!(rejected, top[i])
            end
        end
        
        # The rejected students proposed to their next top choice
        for m in rejected
            cp[m]+=1
            if !(cp[m] in keys(SC.pref[m])) # i.e. the current preference is not in the preference list, and therefore m has no more preferences
                is_matched[m] = true # we assume she is matched to herself
            end
        end

        stop = all(is_matched[i] for i in SC.students) # this boolean tell us if all agents are matched; if so we stop
        if stop || isempty(rejected)
            break
        end
    end

    for m in SC.students
        if is_matched[m]
            continue
        end
        matching[m] = SC.pref[m][cp[m]]
        is_matched[m] = true
    end

    return matching
end


    end
    return frequency_dict
end
