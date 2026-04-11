--- Function description
--- @param emails table
--- @return string
function last_email(emails)
    if not emails then error("Invalid input") end
    local last_email = ""
    for index, email in ipairs(emails) do
        last_email = email
    end
    return last_email
end

-- TESTS
local test1 = { "john@example.com", "jane@example.com" }
assert(last_email(test1) == "jane@example.com", "Test failed: get the last email address in a table")

local test2 = { "john@example.com" }
assert(last_email(test2) == "john@example.com", "Test failed: get the only email address in a table")

local test3 = { "john@example.com", "jane@example.com", "jim@example.com" }
assert(last_email(test3) == "jim@example.com", "Test failed: get the last email address in a longer table")