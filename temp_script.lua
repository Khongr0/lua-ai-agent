function getEvenNumbers(numbers)
    local evenNumbers = {}
    for i, number in ipairs(numbers) do
        if number % 2 == 0 then
            table.insert(evenNumbers, number)
        end
    end
    return evenNumbers
end
